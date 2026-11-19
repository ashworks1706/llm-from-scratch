import axios from "axios";
import {
  PRODUCT_CREATE_REVIEW_FAIL,
  PRODUCT_CREATE_REVIEW_REQUEST,
  PRODUCT_CREATE_REVIEW_SUCCESS,
  PRODUCT_DETAILS_FAIL,
  PRODUCT_DETAILS_REQUEST,
  PRODUCT_DETAILS_SUCCESS,
  PRODUCT_LIST_FAIL,
  PRODUCT_LIST_REQUEST,
  EDIT_REVIEW_REQUEST,
  EDIT_REVIEW_SUCCESS,
  EDIT_REVIEW_RESET,
  PRODUCT_LIST_SUCCESS,
  EDIT_REVIEW_FAIL,
} from "../Constants/ProductConstants";
import URL from "../Url"
import {logout } from "./userActions"
export const listProduct = (keyword=" ") => async (dispatch) => {
  try {
    dispatch({ type: PRODUCT_LIST_REQUEST });
    const { data } = await axios.get(`${URL}/api/products?keyword=${keyword}`);
    dispatch({ type: PRODUCT_LIST_SUCCESS, payload: data });
  } catch (error) {
    dispatch({
      type: PRODUCT_LIST_FAIL,
      payload:
        error.response && error.response.data.message
          ? error.response.data.message
          : error.message,
    });
  }
};


// SINGLE PRODUCT
export const listProductDetails = (id) => async (dispatch) => {
  try {
    dispatch({ type: PRODUCT_DETAILS_REQUEST });
    const { data } = await axios.get(`${URL}/api/products/${id}`);
    dispatch({ type: PRODUCT_DETAILS_SUCCESS, payload: data });
  } catch (error) {
    dispatch({
      type: PRODUCT_DETAILS_FAIL,
      payload:
        error.response && error.response.data.message
          ? error.response.data.message
          : error.message,
    });
  }
};
// SINGLE revuiews
export const createProductReview =
  (productId, review) => async (dispatch, getState) => {
    try {
      dispatch({ type: PRODUCT_CREATE_REVIEW_REQUEST });

      const {
        userLogin: { userInfo },
      } = getState();
      const config = {
        headers: {
          "Content-Type": "application/json", 
          Authorization: `Bearer ${userInfo.token}`,
        },
      };

      await axios.post(
        `${URL}/api/products/${productId}/review`,
        review,
        config
      );
      dispatch({ type: PRODUCT_CREATE_REVIEW_SUCCESS });
    } catch (error) {
      const message =
        error.response && error.response.data.message
          ? error.response.data.message
          : error.message;
      if (message === "Not authorized, token failed") {
        dispatch(logout());
      }
      dispatch({
        type: PRODUCT_CREATE_REVIEW_FAIL,
        payload: message,
      });
    }
  };


  // export const editReview = (productId, review) => async (dispatch) => {
  //   try {
  //     dispatch({ type: EDIT_REVIEW_REQUEST });
      
  //     const { data } = await axios.put(`${URL}/api/products/${productId}/editreview`, review);
  //     dispatch({ type: EDIT_REVIEW_SUCCESS, payload: data });
  
  //     localStorage.setItem("productInfo", JSON.stringify(data));
  //   } catch (error) {
  //     const message =
  //       error.response && error.response.data.message
  //         ? error.response.data.message
  //         : error.message;
  //     dispatch({
  //       type: EDIT_REVIEW_FAIL,
  //       payload: message,
  //     });
  //   }
  // };
  