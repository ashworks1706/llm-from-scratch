import React, { useEffect } from "react";
import { useDispatch } from "react-redux";
import Message from "../components/LoadingError/Error";
import Loader from "../components/LoadingError/Loader";
import { useSelector } from "react-redux";
import { getOrderDetails, payOrder } from "../Redux/Actions/OrderActions";
import moment from "moment";
import { Link } from "react-router-dom";
import GooglePayButton from "@google-pay/button-react";
import { ORDER_PAY_RESET } from "../Redux/Constants/orderConstants";

const OrderScreen = ({ match }) => {
  const orderId = match.params.id;
  window.scrollTo(0, 0);

  const dispatch = useDispatch();

  const orderDetails = useSelector((state) => state.orderDetails);
  const { order, loading, error } = orderDetails;
  const orderPay = useSelector((state) => state.orderPay);
  const { success: successPay } = orderPay;

  if (!loading) {
    const addDecimals = (num) => {
      return (Math.round(num * 100) / 100).toFixed(2);
    };

    order.itemsPrice = addDecimals(
      order.orderItems.reduce((acc, item) => acc + item.price * item.qty, 0)
    );
  }
  useEffect(() => {
    const GooglePayScript = async () => {};
    if (!order || successPay) {
      dispatch({ type: ORDER_PAY_RESET });
      dispatch(getOrderDetails(orderId));
    } else if (!order.isPaid) {
      GooglePayScript();
    }
  }, [dispatch, orderId, successPay, order]);

  const successPaymentHandler = (paymentResult) => {
    dispatch(payOrder, paymentResult);
  };
  return (
    <div className="mx-auto rounded-xl mt-5 bg-wheat p-5 h-[90vh] max-w-8xl">
      {loading ? (
        <Loader />
      ) : error ? (
        <Message variant="alert-danger">{error}</Message>
      ) : (
        <>
          <div className="mt-5 ">
            <h2 className="text-5xl text-center font-medium mb-4">
              Order Confirmation
            </h2>
            <p className="text-2xl font-medium mb-4 ">
              Shipping : {order.shippingAddress.country}
            </p>
            <p className="text-2xl font-medium mb-4 ">
              Pay Method : {order.paymentMethod}
            </p>
          </div>
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
              <tr key={order.id}>
                <td className="p-2 text-md border-t border-gray-400">
                  {order._id}
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
            </tbody>
          </table>
          {!order.isPaid && (
            <div>
              <GooglePayButton
                environment="TEST"
                paymentRequest={{
                  apiVersion: 2,
                  apiVersionMinor: 0,
                  allowedPaymentMethods: [
                    {
                      type: "CARD",
                      parameters: {
                        allowedAuthMethods: ["PAN_ONLY", "CRYPTOGRAM_3DS"],
                        allowedCardNetworks: ["MASTERCARD", "VISA"],
                      },
                      tokenizationSpecification: {
                        type: "PAYMENT_GATEWAY",
                        parameters: {
                          gateway: process.env.REACT_APP_GOOGLE_PAY_GATEWAY || "example",
                          gatewayMerchantId: process.env.REACT_APP_GOOGLE_PAY_GATEWAY_MERCHANT_ID || "exampleGatewayMerchantId",
                        },
                      },
                    },
                  ],
                  merchantInfo: {
                    merchantId: process.env.REACT_APP_GOOGLE_PAY_MERCHANT_ID || undefined,
                    merchantName: "Books of Gold Demo",
                  },
                  transactionInfo: {
                    totalPriceStatus: "FINAL",
                    totalPriceLabel: "Total",
                    totalPrice: order.totalPrice.toString(),
                    currencyCode: "USD",
                    countryCode: "US",
                  },
                }}
                onLoadPaymentData={successPaymentHandler}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
};
export default OrderScreen;
