import React from 'react'
import { useState } from 'react';

const Search = ({history}) => {
    const [keyword, setKeyword] = useState("");

    const searchSubmitHandler = (e) => {
      e.preventDefault();
      if (keyword.trim()) {
        history.push(`/search/${keyword}`);
      } else {
        history.push("/");
      }
    };
  
    return (
      <>
      <div className="flex h-[90vh] flex-col items-center mx-auto my-auto justify-center ">

        <h1 className="text-center p-8 text-3xl" > Search Books</h1>
        <form className="" onSubmit={searchSubmitHandler}>
          <input
            type="text"
            placeholder="Search a Product ..."
            onChange={(e) => setKeyword(e.target.value)}
          />
          <button  type="submit" className='btn'>Search</button>
        </form>
      </div>
      </>)
}

export default Search