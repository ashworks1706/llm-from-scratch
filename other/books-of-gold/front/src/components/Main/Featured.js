import React from "react";
import { Link } from "react-router-dom";
import Rating from "./Rating";
import axios from "axios";
import { useEffect, useState } from "react";
import URL from "../../Redux/Url"


const Featured = () => {
  window.scrollTo(0, 0);

  const [products, setProducts] = useState([]);

  useEffect(() => {
    const fetchproducts = async () => {
      const { data } = await axios.get(`${URL}/api/products`);
      setProducts(data);
    };
    fetchproducts();
  }, []);
  return (
    <div className="flex p-5  flex-wrap flex-col">
      <section>
        <h1 className="aight text-5xl text-center  font-bold ">
          Featured Books
        </h1>
        <p className="text-2xl text-center  mt-3 ">Best of Summer</p>
        <div className="justify-center  items-center flex flex-wrap flex-row">
          {products.map((product) => (
            <div className="p-8  product-card" key={product._id}>
              <div className="">
                <Link to={`/products/${product._id}`}>
                  <img
                    src={product.image}
                    className="max-w-sm"
                    alt={product.name}
                  />
                </Link>
              </div>
              <div className="pt-2">
                <p>
                  <Link
                    to={`/products/${product._id}`}
                    className="text-2xl  font-semibold"
                  >
                    {product.name}
                  </Link>
                </p>
                <Rating
                  value={product.rating}
                  text={`${product.numReviews} reviews`}
                ></Rating>
                <h3 className="text-3xl pt-2 text-semibold text-green-600">
                  ${product.price}
                </h3>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};

export default Featured;
