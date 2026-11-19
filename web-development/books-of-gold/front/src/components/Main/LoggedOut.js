import React from 'react'
import { Link } from "react-router-dom";

const LoggedOut = () => {
  return (
    <div className="flex flex-wrap flex-col p-8 justify-center items-center">
      <h1 className="text-3xl py-5 font-sans">Login to Review</h1>
      <Link to="/login">
        <button className="btn">Login</button>
      </Link>
    </div>
  )
}

export default LoggedOut