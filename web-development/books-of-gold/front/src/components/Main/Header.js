import React from "react";
import Featured from "./Featured";
import { Link } from "react-router-dom";
const Home = () => {
  window.scrollTo(0, 0);

  return (
    <>
      <head title={"Books of Gold | E Commerce Store"} />
      <section id="hero">
        <h4>Editor's Choice</h4>
        <h2>Quality Deals</h2>
        <h1 className="font-semibold">On all products</h1>
        <p>Get Upto 50% off!</p>
        <Link to="/shop">
          <button>Shop Now</button>
        </Link>
      </section>

      <section id="feature" className="section-p1">
        <div className="fe-box">
          <img src="https://vectorx009.github.io/Ecom-Model/free.png" alt="" />
          <h6>Free Delivery</h6>
        </div>
        <div className="fe-box">
          <img
            src="https://vectorx009.github.io/Ecom-Model/refund.png"
            alt=""
          />
          <h6>Guranteed Refund</h6>
        </div>
        <div className="fe-box">
          <img src="https://vectorx009.github.io/Ecom-Model/care.jpg" alt="" />
          <h6>Customer Care</h6>
        </div>
        <div className="fe-box">
          <img src="https://vectorx009.github.io/Ecom-Model/track.png" alt="" />
          <h6>24/7 Tracking</h6>
        </div>
        <div className="fe-box">
          <img
            src="https://vectorx009.github.io/Ecom-Model/secure.png"
            alt=""
          />
          <h6>Secured Transcations</h6>
        </div>
      </section>

      <Featured />
    </>
  );
};

export default Home;
