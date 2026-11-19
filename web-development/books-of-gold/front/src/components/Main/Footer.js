import React from "react";
import logo from "../../data/img/logo.jpg";
const Footer = () => {
  return (
    <>
      <footer className="p-4  footer bg-wheat rounded-lg shadow md:px-6 md:py-8 ">
        <div className="sm:flex sm:items-center sm:justify-between">
          <a href="/" className="flex items-center mb-4 sm:mb-0">
            <img src={logo} className="h-8 mr-3" alt="Flowbite Logo" />
            <span className="self-center text-2xl font-semibold whitespace-nowrap dark:text-white">
              Books of Gold
            </span>
          </a>
          <ul className="flex flex-wrap items-center mb-6 text-md text-gray-500 sm:mb-0 ">
            <li>
              <a href="/" className="mr-4 hover:underline md:mr-6 ">
                Home
              </a>
            </li>
            <li>
              <a href="/shop" className="mr-4 hover:underline md:mr-6">
                Shop
              </a>
            </li>
            <li>
              <a href="/contact" className=" mr-4 hover:underline md:mr-6">
                Contact
              </a>
            </li>
            <li>
              <a href="/about" className="mr-4 hover:underline md:mr-6 ">
                About
              </a>
            </li>
            <li>
              <a href="/profile" className=" mr-4 hover:underline md:mr-6">
                Account
              </a>
            </li>
            <li>
              <a href="/cart/:id?" className=" mr-4  md:mr-6 hover:underline">
                Cart
              </a>
            </li>
          </ul>
        </div>
        {/* <hr className="my-6 border-gray-200 sm:mx-auto dark:border-gray-700 lg:my-8" /> */}
        <hr className="my-6 dark:border-gray-700" />
        <span className="block text-md text-gray-500 sm:text-center dark:text-gray-400">
          ©{" "}
          <a href="https://thevector.xyz" className="hover:underline">
            Som™
          </a>
          . All Rights Reserved.
        </span>
      </footer>
    </>
  );
};

export default Footer;
