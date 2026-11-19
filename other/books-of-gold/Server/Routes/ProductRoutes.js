import express from "express";
import protect from "./../Middleware/AuthMiddleware.js"
import asyncHandler from "express-async-handler";
import Product from "../Models/ProductModel.js";
import User from "../Models/UserModel.js";


const productRoute = express.Router();
// GET ALL PRODUCT
productRoute.get(
  "/",
  asyncHandler(async (req, res) => {
    const keyword = req.query.keyword ? {
      name : {
        $regex: req.query.keyword,
        $options: "i",
      }
    } : {}
    const products = await Product.find({...keyword});
    res.json(products);
  })
);
// GET SINGLE PRODUCT
productRoute.get(
  "/:id",
  asyncHandler(async (req, res) => {
    const product = await Product.findById(req.params.id);
    if (product) {
      res.json(product);
      console.log(product)
    } else {
      res.status(404);
      throw new Error("Product not found");
    }
  })
);
// review SINGLE PRODUCT
productRoute.post(
  "/:id/review",
  protect,
  asyncHandler(async (req, res) => {
    const { rating, comment } = req.body;
    const product = await Product.findById(req.params.id);
    if (product) {
      const alreadyReviewed = product.reviews.find(
        (r) => r.user.toString() === req.user._id.toString()
        );
      if (alreadyReviewed) {
        res.status(404);
        throw new Error("Product already reviewed");
      }
      const review = {
        name: req.user.name,
        rating: Number(rating),
        comment: comment,
        user: req.user._id,
      };
      product.reviews.push(review);
      product.numReviews = product.reviews.length;
      console.log("Submited review")
      product.rating =
        product.reviews.reduce((acc, item) => item.rating + acc, 0) /
        product.reviews.length;
      await product.save();
      res.status(201).json({ message: "Review Added" });
    } else {
      res.status(404);
      throw new Error("Product not found");
    }
  })
);


// productRoute.put(
//   "/:id/editreview",
//   protect,
//   asyncHandler(async (req, res) => {
//     console.log("hello")
//     const product = await Product.findById(req.params.id);
//     const user = await User.findById(req.user._id);


//     if (product.reviews.user === user) {
//       product.reviews.rating = req.body.rating;
//       product.reviews.comment = req.body.comment;
//       const updatedReview = await product.save();

//       res.json({
//         rating: updatedReview.rating,
//         comment: updatedReview.comment,
//       });
//     } else {
//       res.status(404);
//       throw new Error("User Review not found");
//     }
//   })
// );



export default productRoute;
