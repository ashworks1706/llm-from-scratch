import React from "react";
import { useSelector } from "react-redux";
import moment from "moment";
import { Link } from "react-router-dom";
import Message from "../LoadingError/Error";
import Loader from "../LoadingError/Loader";
import EmptyOrders from "../Main/EmptyOrders";
const Orders = () => {
  window.scrollTo(0, 0);
  const userLogin = useSelector((state) => state.userLogin);
  const { error, loading } = userLogin;
  const orderListMy = useSelector((state) => state.orderListMy);
  const { orders } = orderListMy;
  return (
    <div className="container mx-auto rounded-xl mt-5 bg-wheat p-5 h-[90vh] max-w-8xl">
      {error && <Message variant={"alert-danger"}>{error}</Message>}
      {loading && <Loader />}
      <div className="table-responsive ">
        {orders.length < 1 ? (
          <EmptyOrders />
        ) : (
          <table className="w-full text-left table-collapse">
            <thead>
              <tr>
                <th className="text-lg font-medium text-gray-700 p-2 bg-gray-200">
                  Order ID
                </th>
                <th className="text-lg font-medium text-gray-700 p-2 bg-gray-200">
                  Date
                </th>
                <th className="text-lg font-medium text-gray-700 p-2 bg-gray-200">
                  Product
                </th>
                <th className="text-lg font-medium text-gray-700 p-2 bg-gray-200">
                  Price
                </th>
                <th className="text-lg font-medium text-gray-700 p-2 bg-gray-200">
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(orders).map((order) => {
                return (
                  <tr key={order.id}>
                    <td className="p-2 text-md border-t border-gray-400">
                      <Link to={`/order/${order.id}`}>{order._id}</Link>
                    </td>
                    <td className="p-2  text-md border-t border-gray-400">
                      {moment(order.createdAt).format("LL")}
                    </td>
                    <td className="p-2  text-md border-t border-gray-400">
                      {order.orderItems.map((a, e) => {
                        return (
                          <Link to={`/products/${order.orderItems[e].product}`}>
                            {order.orderItems[e].name} <br />
                          </Link>
                        );
                      })}
                    </td>
                    <td className="p-2   text-md border-t border-gray-400">
                      ${order.totalPrice}
                    </td>
                    <td
                      className={`${
                        order.isDelivered ? "bg-green" : "bg-white"
                      } p-2 text-md border-t border-gray-400`}
                    >
                      {order.isPaid
                        ? "Order Completed"
                        : order.isDelivered
                        ? "Order has been delivered "
                        : "In Transit"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default Orders;
