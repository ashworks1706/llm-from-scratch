import React, { useEffect, useState } from "react";
import Rating from "../components/Main/Rating";
import { useDispatch, useSelector } from "react-redux";
import { createProductReview, listProductDetails,editReview } from "../Redux/Actions/ProductActions";
import Loader from "../components/LoadingError/Loader";
import moment from "moment";
import LoggedOut from "../components/Main/LoggedOut";
import Error from "../components/LoadingError/Error";
import StockErrored from "../components/Main/StockError";
import { PRODUCT_CREATE_REVIEW_RESET } from "../Redux/Constants/ProductConstants";
import Message from "../components/LoadingError/Error";

const ProductDetailsPage = ({ history, match }) => {
  const [qty, setQty] = useState(1);
  const [edit, setedit] = useState(false)
  const [rating, setRating] = useState(0);
  const [comment, setComment] = useState("");
  const [editrating, setEditrating] = useState(0);
  const [editcomment, setEditcomment] = useState("");
  const productId = match.params.id;

  const dispatch = useDispatch();
  const userLogin = useSelector((state) => state.userLogin);
  const { userInfo } = userLogin;

  const productDetails = useSelector((state) => state.productDetails);
  const { loading, error, product } = productDetails;
  const productReview = useSelector((state) => state.productReview);
  const {
    loading: loadingCreateReview,
    error: errorCreateReview,
    success: successCreateReview,
  } = productReview; 

  useEffect(() => {
    if (successCreateReview) {
      alert("Review Submitted");
      setRating(0);
      setComment("");
      dispatch({ type: { PRODUCT_CREATE_REVIEW_RESET } });
    }
    dispatch(listProductDetails(productId));
  }, [dispatch, productId, successCreateReview]);

  const handleAddToCart = (e) => {
    e.preventDefault();
    history.push(`/cart/${productId}?qty=${qty}`);
  };

  const handleSubmit =(e) =>{
    e.preventDefault();
    dispatch(createProductReview(productId, {rating, comment}))
  }
  const SubmitEdit =(e) =>{
    e.preventDefault();
    window.location.href="/"
    // dispatch(editReview(productId,{  id: userLogin._id, editrating, editcomment, }))
  }
  
  const handledit =()=>{
    setedit(true)
  }
  const cancledit=() =>{
    setedit(false)
  }
  return (
    <>
      <section className="flex-row  items-center justify-center flex flex-wrap p-8 ">
        {loading ? (
          <Loader />
        ) : error ? (
          <Error />
        ) : (
          <>
            <div className="p-8">
              <img
                src={product.image}
                alt={product.name}
                className="max-w-md mb-4 rounded-lg"
              />
            </div>
            <div>
              {product.countInStock === 0 ? <StockErrored /> : ""}

              <h2 className="text-4xl font-semibold mb-4">{product.name}</h2>
              <h3 className="text-xl  mb-4">{product.author}</h3>
              <p className="text-gray-700 text-sans max-w-lg mb-4">
                {product.description}
              </p>
              <h3 className="text-3xl pt-2 font-bold text-green-600">
                ${product.price}
              </h3>

              {product.countInStock > 0 ? (
                <>
                  <h3 className="text-2xl pt-2 font-semibold text-green-600">
                    In Stock
                  </h3>
                  <div className="text-2xl mb-6">
                    <div>
                      <button
                        className={`${
                          qty < 0 || qty === 1
                            ? "cursor-not-allowed opacity-50"
                            : "text-red-700"
                        } pt-3  py-2 px-4 rounded-lg  mr-2`}
                        onClick={
                          qty < 0 || qty === 0
                            ? setQty(1)
                            : () => setQty(qty - 1)
                        }
                      >
                        -
                      </button>
                      <span value={qty} className="text-yellow-500 font-medium">
                        {qty}
                      </span>
                      <button
                        className={`
                    pt-3  py-2 px-4 rounded-lg  mr-2`}
                        onClick={() => setQty(qty + 1)}
                      >
                        +
                      </button>
                      <button
                        className={`py-2 px-4 rounded-lg ml-4 btn`}
                        onClick={handleAddToCart}
                        disabled={qty > product.countInStock ? true : false}
                      >
                        Add to Cart
                      </button>
                    </div>
                  </div>
                  {qty > product.countInStock ? (
                    <div className="alert">
                      <h1>Maximum qty Exceeded</h1>
                    </div>
                  ) : null}
                  <div className="text-gray-700">
                    <p className="mb-2 text-2xl ">Reviews</p>
                    <ul className="list-inside">
                      <Rating
                        value={product.rating}
                        text={`${product.numReviews} reviews`}
                      />
                    </ul>
                  </div>
                </>
              ) : (
                <>
                  <h3 className="text-2xl pt-2 font-semibold text-red-600 ">
                    <strike>Out of Stock</strike>
                  </h3>
                </>
              )}
            </div>
          </>
        )}
      </section>
        <div className="mt-10">
      <h2 className="text-6xl text-center font-bold my-5"> Reviews</h2>
      { product.reviews.length === 0 &&(
        <div className=" text-3xl text-center">
          No Reviews
        </div>
      )}
      {product.reviews.map(review => (
        <div key={review.id} className="bg-wheat min-h[50vh] p-8 mb-5 rounded-lg">
        <div className="flex  justify-between">
          <div className="flex items-center mb-3">
              <h1 className="text-2xl font-bold">{review.name}</h1>
              <Rating value={review.rating} text={review.rating}/>
              <p className="text-xl text-gray-600">{moment(review.createdAt).format("LL")}</p>
            </div>
            {review.name === userInfo.name?
            <div className="edit" >
            <i class={ edit===false ?"text-3xl fa-solid fa-pen-to-square": "text-3xl fa-sharp fa-solid fa-xmark"} id="pencil" onClick={ edit===false? handledit:cancledit}></i>
          </div>
            :""
            }
        </div>
          <p className="text-md">{review.comment}</p>
        </div>
      ))}
    </div>


    <div className="flex flex-col items-center py-4">
      <h2 className="text-lg font-bold">Leave a Review</h2>
      <div className="my-4">
        {loadingCreateReview && <Loader/>}
        {errorCreateReview && <Message variant={"alert-danger"}>{error}</Message>}
      </div>
      {userInfo?
      edit ===false? 
      <form onSubmit={handleSubmit}>
        
        <div className="mb-4">
          <label
            className="block text-xl text-gray-700 font-medium mb-2"
            htmlFor="review"
          >
            Review:
          </label>
          <textarea
            className="w-full border border-gray-400 p-2"
            id="review"
            rows="4"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
          />
        </div>
        <div className="mb-4">
          <label
            className="block text-gray-700 font-medium mb-2"
            htmlFor="rating"
          >
            Rating:
          </label>
          <select
            className="w-full border border-gray-400 p-2"
            id="rating"
            value={rating}
            onChange={(e) => setRating(e.target.value)}
          >
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
            <option value="5">5</option>
          </select>
        </div>
        <button
          className="btn"
          type="submit"
        >
          Submit Review
        </button>
      </form> : 
      <form onSubmit={SubmitEdit}>
        
        <div className="mb-4">
          <label
            className="block text-xl text-gray-700 font-medium mb-2"
            htmlFor="review"
          >
            Edit Your Review:
          </label>
          <textarea
            className="w-full border border-gray-400 p-2"
            id="review"
            rows="4"
            value={editcomment}
            onChange={(e) => setEditcomment(e.target.value)}
          />
        </div>
        <div className="mb-4">
          <label
            className="block text-gray-700 font-medium mb-2"
            htmlFor="rating"
          >
            Edit Rating:
          </label>
          <select
            className="w-full border border-gray-400 p-2"
            id="rating"
            value={editrating}
            onChange={(e) => setEditrating(e.target.value)}
          >
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
            <option value="5">5</option>
          </select>
        </div>
        <button
          className="btn"
          type="submit"
        >
          Update Review
        </button>
      </form>
       : <LoggedOut/>
      }
    </div>
    </>
  );
};

export default ProductDetailsPage;
