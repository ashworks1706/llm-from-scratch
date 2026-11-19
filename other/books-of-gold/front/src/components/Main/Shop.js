import Rating from "./Rating";
import { Link } from "react-router-dom";
import React, { useEffect } from "react";
import ShopHeader from "./ShopHeader";
import { useDispatch, useSelector } from "react-redux";
import { listProduct } from "../../Redux/Actions/ProductActions";
import Loader from "../LoadingError/Loader.js";
import Message from "../LoadingError/Error.js";

const Shop = ({match}) => {
  window.scrollTo(0, 0);

  const dispatch = useDispatch();
  const keyword = match.params.keyword 
  const productList = useSelector((state) => state.productList);
  const { loading, error, products } = productList;

  useEffect(() => {

    dispatch(listProduct(keyword));
  }, [dispatch,keyword]);

  return (
    <>
      {loading ? (
        <Loader />
      ) : error ? (
        <Message variant={"alert-danger"}>{error}</Message>
      ) : (
        <>
          <ShopHeader />
          <div className="flex flex-wrap flex-row ">
            {products.map((product) => (
              <div className="p-8 w-1/3 product-card" key={product._id}>
                <div>
                  <Link to={`/products/${product._id}`}>
                    <img
                      src={product.image}
                      className="w-full"
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
                      {product.name}{" "}
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
        </>
      )}
    </>
  );
};

export default Shop;
