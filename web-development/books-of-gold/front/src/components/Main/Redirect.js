import React from "react";

const Redirect = () => {
  window.scrollTo(0, 0);

  setTimeout(() => {
    window.location.href = "http://thevector.xyz";
  }, 500);
  return (
    <div className=" load text-center text-3xl p-8">
      Redirecting you to Som's Profile ...
    </div>
  );
};

export default Redirect;
