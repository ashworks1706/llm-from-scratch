import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Link } from "react-router-dom";
import Message from "../components/LoadingError/Error";
import Loader from "../components/LoadingError/Loader";
import { register } from "../Redux/Actions/userActions";

const RegisterPage = ({ location, history }) => {
  const [email, setEmail] = useState("");
  window.scrollTo(0, 0);

  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");

  const dispatch = useDispatch();
  const redirect = location.search ? location.search.split("=")[1] : "/";
  const userRegister = useSelector((state) => state.userRegister);
  const { error, loading, userInfo } = userRegister;

  useEffect(() => {
    if (userInfo) {
      history.push(redirect);
    }
  }, [userInfo, history, redirect]);

  const handleSubmit = (e) => {
    e.preventDefault();
    dispatch(register(name, email, password));
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      {error && <Message variant={"alert-danger"}>{error}</Message>}
      {loading && <Loader />}

      <form
        className="bg-wheat p-6 rounded-lg shadow-md"
        onSubmit={handleSubmit}
      >
        <h2 className="text-2xl font-medium mb-4">Create an Account</h2>
        <div className="mb-4">
          <label className="block text-gray-700 font-medium mb-2">Name</label>
          <input
            className="border border-gray-400 p-2 w-full"
            type="email"
            placeholder="example@email.com"
            name="email"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        <div className="mb-4">
          <label className="block text-gray-700 font-medium mb-2">Email</label>
          <input
            className="border border-gray-400 p-2 w-full"
            type="email"
            placeholder="example@email.com"
            name="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
        </div>
        <div className="mb-4">
          <label className="block text-gray-700 font-medium mb-2">
            Password
          </label>
          <input
            className="border border-gray-400 p-2 w-full"
            type="password"
            placeholder="********"
            name="password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
          />
        </div>
        <div className="mb-4">
          <label className="block text-gray-700 font-medium mb-2">
            Confirm Password
          </label>
          <input
            className="border border-gray-400 p-2 w-full"
            type="password"
            placeholder="********"
            name="confirmPassword"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </div>
        <div className="mb-4 font-semibold font-sans ">
          {password !== confirm && password !== 0 && password !== "" ? (
            <h2>Confirmed password is not the same</h2>
          ) : (
            "Welcome!"
          )}
        </div>
        <button
          type="submit"
          disabled={password === confirm ? false : true}
          className="btn"
        >
          Create Account
        </button>
        Already have an account?
        <Link to={redirect ? `/login?redirect=${redirect}` : "/login"}>
          <button className="btn">Login</button>
        </Link>
      </form>
    </div>
  );
};

export default RegisterPage;
