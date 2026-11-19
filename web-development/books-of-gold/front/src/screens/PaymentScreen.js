import React from "react";
import { useDispatch, useSelector } from "react-redux";
import { savePaymentMethod } from "../Redux/Actions/cartActions";

const PaymentScreen = ({ history }) => {
  window.scrollTo(0, 0);

  const dispatch = useDispatch();
  const cart = useSelector((state) => state.cart);
  const { shippingAddress } = cart;

  if (!shippingAddress) {
    history.push("/shipping");
  }

  const paymentMethod = ("GooglePay");
  const handleSubmit = (event) => {
    event.preventDefault();
    dispatch(savePaymentMethod(paymentMethod));
    history.push("/placeorder");
  };

  return (
    <div className="mx-auto payment-screen p-8 flex flex-col items-center justify-center p-4">
      <div className="p-4 text-center  bg-white rounded-lg shadow-md">
        <h1 className="text-xl font-medium">Select Payment Method</h1>
        <div className="my-4">
          <form className="payment" onSubmit={handleSubmit}>
            <button type="submit" className="btn">
              Google Pay
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default PaymentScreen;
