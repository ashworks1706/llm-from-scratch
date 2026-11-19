import React from "react";
import { Link } from "react-router-dom";

const EmptyOrders = () => {
  return (
    <div className="flex flex-wrap flex-col p-8 justify-center items-center">
      <h1 className="text-3xl py-5 font-sans">You have No Orders!</h1>
      <Link to="/shop">
        <button className="btn">Start Shopping</button>
      </Link>
    </div>
  );
};

export default EmptyOrders;
