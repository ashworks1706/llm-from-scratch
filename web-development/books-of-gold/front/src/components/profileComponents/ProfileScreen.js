import React, { useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { Link } from "react-router-dom";
import {
  getUserDetails,
  logout,
  updateUserProfile,
} from "../../Redux/Actions/userActions";
import { toast, ToastContainer } from "react-toastify";
import moment from "moment";
import Message from "../LoadingError/Error";
import Loader from "../LoadingError/Loader";
import { listOrder } from "../../Redux/Actions/OrderActions";
const ProfileScreen = () => {
  const dispatch = useDispatch();

  const userLogin = useSelector((state) => state.userLogin);
  const { error, loading, userInfo } = userLogin;
  const orderListMy = useSelector((state) => state.orderListMy);
  const {orders } = orderListMy;
  const userDetails = useSelector((state) => state.userDetails);
  const { user } = userDetails;
  
  const userUpdateProfile = useSelector((state) => state.userUpdateProfile);
  const { loading: updateLoading } = userUpdateProfile;

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const toastId = React.useRef(null);
  const ToastObjects = {
    pauseOnFocusLoss: false,
    draggable: false,
    pauseOnHover: false,
    autoClose: 2000,
  };
  const handlelogout = (event) => {
    event.preventDefault();
    dispatch(logout());
    if (!toast.isActive(toastId.current)) {
      toastId.current = toast.success("Profile Updated", ToastObjects);
    }
  };

  useEffect(() => {
    dispatch(listOrder());
    dispatch(getUserDetails("profile"));
    if (user) {
      setName(user.name);
      setEmail(user.email);
    }
  }, [dispatch, user]);
 
  const handleSubmit = (event) => {
    event.preventDefault();
    if (password !== confirm) {
      if (!toast.isActive(toastId.current)) {
        toastId.current = toast.error("Password does not match", ToastObjects);
      }
    } else {
      dispatch(updateUserProfile({ id: user._id, name, email, password }));
    }
  };
  return (
    <>
      <ToastContainer
        position="top-right"
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
      />
      {error && <Message variant={"alert-danger"}>{error}</Message>}
      {loading && <Loader />}
      {updateLoading && <Loader />}
      <div className="container mx-auto p-5">
        <h2 className="text-6xl text-center font-medium my-8">Your Profile</h2>
        {userInfo ? (
          <>
            <div className="flex-row flex  bg-wheat flex-wrap justify-between items-center">
              <div className="p-4 mx-auto items-center flex flex-col flex-wrap">
                <div className="p-5  text-6xl">
                  <strong >Hi {(userInfo.name).toUpperCase()}</strong>
                </div>
                <div className="p-5 text-2xl">
                  Joined {moment(userInfo.createdAt).format("LL")}
                </div>
                <div className="text-2xl">
                  <Link to={"/orders"}>
                  <button className="btn">Your Orders
                  <span className="mx-2">{orders ? orders.length : 0}</span>
                   </button>
                  </Link>
                </div>
              </div>
              <form
                className="bg-wheat p-6 rounded-lg shadow-md"
                onSubmit={handleSubmit}
              >
                <div className="mb-4">
                  <label className="block text-gray-700 font-medium mb-2">
                    Name
                  </label>
                  <input
                    className="border border-gray-400 p-2 rounded-lg w-full"
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                </div>
                <div className="mb-4">
                  <label className="block text-gray-700 font-medium mb-2">
                    Email
                  </label>
                  <input
                    className="border border-gray-400 p-2 rounded-lg w-full"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
                <div className="mb-4">
                  <label className="block text-gray-700 font-medium mb-2">
                    New Password
                  </label>
                  <input
                    className="border border-gray-400 p-2 rounded-lg w-full"
                    type="password"
                    value={confirm}
                    onChange={(e) => setConfirm(e.target.value)}
                    required
                  />
                </div>
                <div className="mb-4">
                  <label className="block text-gray-700 font-medium mb-2">
                    Confirm Password
                  </label>
                  <input
                    className="border border-gray-400 p-2 rounded-lg w-full"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
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
                  Update Profile
                </button>
                <Link onClick={handlelogout}>
                  <button className="p-3 ">Logout</button>
                </Link>
              </form>
            </div>
          </>
        ) : (
          <div className="container">
            <Link to={"/login"}>Login</Link>
            <Link to={"/register"}>Register</Link>
          </div>
        )}
      </div>
    </>
  );
};

export default ProfileScreen;
