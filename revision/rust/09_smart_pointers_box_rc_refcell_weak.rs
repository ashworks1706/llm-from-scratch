// Box<T> owns one heap value; its size is a pointer, which is what makes recursive types possible.
// Rc<T> is shared ownership on one thread; the value drops when the strong count hits 0.
// RefCell<T> moves borrow checking to runtime: a conflicting borrow_mut panics, try_borrow_mut returns Err.
// Rc<RefCell<T>> parent <-> child links form a cycle that never drops; Weak<T> does not count toward ownership.
// Weak::upgrade returns Option<Rc<T>>: None once every strong owner is gone.
// from piramid: apps/engine/serving/src/machine.rs holds Arc<RwLock<Latest>> and gives the sampler thread only a Weak,
// so the thread stops when the last MachineReadings clone drops. reproduce the shape single-threaded with Rc.
// explain: why does a parent -> child strong link plus child -> parent strong link leak, and which direction gets the Weak?
// explain: RefCell compiles a double borrow_mut and panics at runtime. what did the borrow checker trade away?

use std::cell::RefCell;
use std::rc::{Rc, Weak};

enum Expr {
    Num(i64),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
}

fn eval(expr: &Expr) -> i64 {
    // recurse through the boxes by reference; no clones, no moves out of the tree.
    todo!()
}

struct TreeNode {
    value: i32,
    parent: RefCell<Weak<TreeNode>>,
    children: RefCell<Vec<Rc<TreeNode>>>,
}

fn add_child(parent: &Rc<TreeNode>, child: Rc<TreeNode>) {
    // push the child into the parent's list and point the child's parent at the parent, weakly.
    // invariant: adding a child must not change the parent's strong count.
    todo!()
}

fn depth(node: &Rc<TreeNode>) -> usize {
    // number of ancestors, walking parent links with upgrade. the root has depth 0.
    todo!()
}

fn second_borrow_fails(cell: &RefCell<i32>) -> bool {
    // hold one borrow_mut, then try another; return true when the second attempt is refused.
    // the first guard must still be alive when you try the second.
    todo!()
}
