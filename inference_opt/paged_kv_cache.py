# so normally in KV cache, we reserve a massive contiguous block of VRAM for every user, -> 4096
# what if the user just says "Hi!" , in that case all 4095 slots of VRAM is unused
# paged attention uses noncontiguous blocks of memory to provide on demand service when needed in VRAM


import torch
import math 

class PhysicalTokenBlock:
    def __init__(self, config):
        # this is a single chunk of physical memory on the GPU
        # this holds the block_size tokens
        
        self.block_number = config.block_number
        self.block_size = config.block_size
        
        # the actual storage
        # shape: (num_heads, block_size, head_dim)
        # We put block_size in the middle for contiguous memory access pattern

        self.key_block = torch.zeros((config.num_heads ,config.block_size, config.head_dim), dtype=torch.float32, device=config.device)
        self.value_block = torch.zeros((config.num_heads ,config.block_size, config.head_dim), dtype=torch.float32, device=config.device)

        self.ref_count = 0 # for beam search / sharing

        # beamsearch is actually just a top k parameter to find a good sweet spot between exhaustive search
        # and greedy search for next word prediction on scale of how confident the llm is choosing the next word

        self.num_filled = 0


# now we will configure it 
class KVBlockManager:
    def __init__(self, config):
        self.block_size = config.block_size
        self.free_blocks = []
        self.all_blocks = []
        # pre alocating all gpu memory at startup

        for i in range(config.num_blocks):
            block = PhysicalTokenBlock(i, config.block_size, config.head_dim, config.num_heads, config.device)
            self.all_blocks.append(block)
            self.free_blocks.append(i)


    def allocate_block(self, request_id):
        # assigning a new free block to the user
        if not self.free_blocks:
            raise MemoryError("OOM: No more free KV Blocks!")

        # popping a free block index
        physical_block_id = self.free_blocks.pop()

        # assigning to user's block table
        if request_id not in self.block_tables:
            self.block_tables[request_id] = []

        self.block_tables[request_id].append(physical_block_id)

        return self.all_blocks[physical_block_id]


    def get_physical_block(self, request_id, logical_token_index):
        # logical index -> physical block 
        # if user wants token #20, and blokc size is 16
        # - it's in the 2nd block (index1)
        # - offset is 4
        
        block_table = self.block_tables[request_id]


        # which block in the sequence is this?
        logical_block_idx = logical_token_index // self.block_size

        # if we need a new bock, allocate it 
        if logical_block_idx >= len(block_tables):
            self.allocate_block(request_id)

        physical_block_id = block_table[logical_block_idx]

        return self.all_blocks[physical_block_id]

    def append_token(self, request_id, key_vector, value_vector):
        # writes a single token's KV to the correct spot in the physical memory
        
        # determining where the next empty slot is 
        if request_id not in self.block_tables:
            self.allocate_block(request_id)

        # get the last allocated block
        last_physical_id = self.block_tables[request_id][-1]
        last_block = self.all_blocks[last_physical_id]
        

        # is this block full?
        if last_block.num_filled>= self.block_size:
            # allocate a new block
            last_block = self.allocate_block(request_id)
        

        # write data
        # key_vector shape : (Num_heads, head_dim)
        slot = last_block.num_filled
        last_block.key_block[:,slot,:] = key_vector
        last_block.value_block[:,slot,:] = value_vector

        last_block.num_filled+=1



    def get_attention_memory(self, request_id):
        """
        Reconstructs the full Key/Value tensors for attention.
        In vLLM, this is done by a custom CUDA kernel that reads directly from blocks.
        Here, we will 'Gather' them back into a contiguous tensor for PyTorch.
        """
        block_ids = self.block_tables[request_id]
        
        keys_list = []
        values_list = []
        
        for physical_id in block_ids:
            block = self.all_blocks[physical_id]
            # We only take the filled portion of the block
            # (Or typically the kernel handles masks, but let's take valid data)
            k = block.key_block[:, :block.num_filled, :]
            v = block.value_block[:, :block.num_filled, :]
            keys_list.append(k)
            values_list.append(v)
            
        # Concatenate along the Sequence dimension (dim=1)
        # Result: (Num_Heads, Total_Seq_Len, Head_Dim)
        full_k = torch.cat(keys_list, dim=1)
        full_v = torch.cat(values_list, dim=1)
        
        return full_k, full_v



