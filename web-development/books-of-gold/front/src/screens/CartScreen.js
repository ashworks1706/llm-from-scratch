import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Link } from "react-router-dom";
import EmptyCart from "../components/Main/EmptyCart";
import { addToCart, removefromcart } from "../Redux/Actions/cartActions";

const CartScreen = ({ match, location, history }) => {
  window.scrollTo(0, 0);
  const productId = match.params.id;
  const qty = location.search ? Number(location.search.split("=")[1]) : 1;

  const dispatch = useDispatch();
  useEffect(() => {
    if (productId) {
      dispatch(addToCart(productId, qty));
    }
  }, [dispatch, productId, qty]);
  const cart = useSelector((state) => state.cart);
  const { cartItems } = cart;

  const total = cartItems.reduce((a, i) => a + i.qty * i.price, 0).toFixed(2);

  const checkoutHandler = () => {
    history.push("/login?redirect=shipping");
  };
  const handleRemove = (id) => {
    dispatch(removefromcart(id));
  };
  return (
    <div className=" p-8 mx-auto">
      <section id="page-header" className="cartheader">
        <h2>Cart</h2>
        <p>Every deal is a deal.</p>
      </section>
      <div>
        {cartItems.length <= 0 ? (
          <EmptyCart />
        ) : (
          <table className="w-full text-center ">
            <thead className="p-5">
              <tr className="text-2xl bg-gray-100">
                <th>Product</th>
                <th>Name</th>
                <th>Price</th>
                <th>Quantity</th>
                <th>Remove</th>
              </tr>
            </thead>
            <tbody>
              {cartItems.map((item) => (
                <tr key={item.id} className="border-b py-5">
                  <td>
                    <img src={item.image} alt={item.name} />
                  </td>
                  <td>
                    <Link to={`/products/${productId}`}>{item.name}</Link>
                  </td>
                  <td className="text-green-500">${item.price}</td>
                  <td>
                    <select
                      className="p-5"
                      value={item.qty}
                      onChange={(e) =>
                        dispatch(
                          addToCart(item.product, Number(e.target.value))
                        )
                      }
                    >
                      {[...Array(item.countInStock).keys()].map((x) => (
                        <option className="p-5" key={x + 1} value={x + 1}>
                          {x + 1}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td>
                    <button
                      className="bg-red-500 text-white rounded-lg p-2"
                      onClick={() => handleRemove(item.product)}
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <div className="my-4 flex flex-wrap flex-col justify-center items-center text-right">
        <h3 className="text-3xl font-medium py-2">TOTAL </h3>
        <h3 className="text-4xl py-2 font-medium text-green-600">${total}</h3>

        {total > 0 && (
          <button className="btn" onClick={checkoutHandler}>
            Checkout
          </button>
        )}
      </div>
    </div>
  );
};

export default CartScreen;
