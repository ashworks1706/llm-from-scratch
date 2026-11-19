import express from "express";
import dotenv from "dotenv";
import connectDatabase from "./config/MongoDb.js";
import productRoute from "./Routes/ProductRoutes.js";
import { errorHandler, notFound } from "./Middleware/Errors.js";
import cors from "cors";
import userRoute from "./Routes/UserRoutes.js";
import orderRouter from "./Routes/orderRoutes.js";

dotenv.config();
const app = express();
app.use(express.json())
const port = Number(process.env.PORT || 5000);
// API
app.use(cors());
app.use( "/api/products", productRoute);
app.use( "/api/users", userRoute);
app.use( "/api/orders", orderRouter);
app.use(notFound);
app.use(errorHandler);

if (!process.env.MONGO_URL || !process.env.JWT_SECRET) {
  throw new Error("Set MONGO_URL and JWT_SECRET before starting the demo.");
}

await connectDatabase();
app.listen(port, "127.0.0.1", () => {
  console.log(`Bookstore API running at http://127.0.0.1:${port}`);
});
