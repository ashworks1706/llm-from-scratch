# order status is a small state machine: pending, processing, shipped, delivered, cancelled.
# cancelled is terminal. entering it has side effects (restock, refund) that must run exactly once.
# refund owed = -(sum of purchase and refund rows for this order). a second cancel owes 0, so it is idempotent.
# deleting an order that is not cancelled or delivered also restocks and refunds.
# from platform: modules/storefront/service.py ORDER_STATUSES, change_order(), _refund(), _restock(), remove_order().
# from platform: events are strings like "Storefront Purchase - Order #7" and "Refund - Order #7".
# explain: why derive the refund from the ledger instead of order.total_amount?
# explain: a pending member order has no purchase row. why does it get no refund?

STATUSES = ["pending", "processing", "shipped", "delivered", "cancelled"]


class StoreError(Exception):
    def __init__(self, message: str, status: int = 400):
        pass


def refund_owed(ledger: list[tuple[str, float]], order_id: int) -> float:
    # ledger rows are (event, points). sum the rows whose event is the purchase or refund event of order_id.
    # return max(0, -sum). exact event strings, so order 7 never matches order 70.
    pass


def change_status(order: dict, new: str, ledger: list[tuple[str, float]], stock: dict[int, int]) -> None:
    # order = {"id", "status", "items": [(product_id, qty)]}. unknown status -> StoreError 400.
    # cancelled -> anything else -> StoreError 409. same status is a no-op.
    # moving to cancelled from another status: add qty back to stock and append a refund row if owed > 0.
    pass


def remove_order(order: dict, ledger: list[tuple[str, float]], stock: dict[int, int]) -> None:
    # delete: restock and refund unless status is cancelled or delivered.
    pass


if __name__ == '__main__':
    # assert: purchase -300 then cancel gives +300 and stock restored, cancelling twice refunds once,
    # cancelled -> shipped raises 409, order 7 events never count for order 70,
    # deleting a delivered order touches neither stock nor ledger.
    pass
