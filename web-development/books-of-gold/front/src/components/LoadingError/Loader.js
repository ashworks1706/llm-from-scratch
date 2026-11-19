import React from "react";
import { Vortex } from "react-loader-spinner";

const Loader = () => {
  const quote = [
    "Be Yourself",
    "Believe in yourself",
    "Do the hardwork especially when you don't feel like it",
    "I keep my goals secret to achieve them faster",
    "Always strive for the best",
    "The Day you stop learning is the day you die",
    "Delayed Gratification",
    "Never Give Up",
    "Never Settle",
    "Great Wall of China was never built in a day",
    "Beyond Fear Lies Victory",
  ];
  return (
    <>
      <div className="load">
        <>
          <h1 className="text-4xl font-serif text-yellow-300">
            <q>{quote[Math.floor(Math.random() * 10) + 1]}</q>
          </h1>
          <Vortex
            visible={true}
            height="80"
            width="80"
            ariaLabel="vortex-loading"
            wrapperStyle={{}}
            wrapperclassName="vortex-wrapper"
            colors={["wheat", "wheat", "wheat", "wheat", "wheat", "wheat"]}
          />
        </>
      </div>
      ;
    </>
  );
};

export default Loader;
