import React, { useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import { saveAddress } from "../Redux/Actions/cartActions";
// import { updateShipping } from './actions'

function ShippingScreen({ history }) {
  const dispatch = useDispatch();
  window.scrollTo(0, 0);

  const cart = useSelector((state) => state.cart);
  const { shippingAddress } = cart;
  const [address, setAddress] = useState(shippingAddress.address);
  const [city, setCity] = useState(shippingAddress.city);
  const [country, setCountry] = useState(shippingAddress.country);
  const [postalCode, setpostalCode] = useState(shippingAddress.postalCode);

  const handleSubmit = (event) => {
    event.preventDefault();
    dispatch(saveAddress({ address, city, postalCode, country }));
    history.push("/payment");
  };

  return (
    <div className="flex items-center justify-center h-[90vh] bg-white p-6 rounded-lg">
      <form onSubmit={handleSubmit}>
        <label className="block font-medium text-lg mb-2">
          Shipping Address
        </label>
        <input
          className="border border-gray-400 p-2 rounded-lg w-full mb-4"
          type="text"
          placeholder="Address"
          value={address}
          required
          onChange={(e) => setAddress(e.target.value)}
        />
        <input
          className="border border-gray-400 p-2 rounded-lg w-full mb-4"
          type="text"
          required
          placeholder="City"
          value={city}
          onChange={(e) => setCity(e.target.value)}
        />
        <input
          className="border border-gray-400 p-2 rounded-lg w-full mb-4"
          type="text"
          required
          placeholder="Country"
          value={country}
          onChange={(e) => setCountry(e.target.value)}
        />
        <input
          className="border border-gray-400 p-2 rounded-lg w-full mb-4"
          type="text"
          required
          placeholder="Postal Code"
          value={postalCode}
          onChange={(e) => setpostalCode(e.target.value)}
        />
        <button className="btn" type="submit">
          Save
        </button>
      </form>
    </div>
  );
}

export default ShippingScreen;
