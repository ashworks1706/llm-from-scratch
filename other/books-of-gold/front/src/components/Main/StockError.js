import React from "react";

const StockError = () => {
  function clearImmediate() {
    const a = document.getElementById("error");
    a.classList.add("hidden");
  }
  window.scrollTo(0, 0);

  return (
    <div
      id="error"
      className="flex text-md justify-between text-black shadow-inner rounded p-3 bg-red-300"
    >
      <p className="self-center">
        <strong> Holy Smokes! </strong>We're Sold Out
      </p>
      <button
        onClick={clearImmediate}
        className="text-xl align-center cursor-pointer alert-del"
      >
        &times;
      </button>
    </div>
  );
};

export default StockError;
