import React from "react";
import { Link } from "react-router-dom";
const EmptyCart = () => {
  window.scrollTo(0, 0);

  return (
    <div className="flex flex-wrap flex-col p-8 justify-center items-center">
      <h1 className="text-3xl py-5 font-sans">Your Cart is Empty!</h1>
      <Link to="/shop">
        <button className="btn">Continue Shopping</button>
      </Link>
    </div>
  );
};

export default EmptyCart;
