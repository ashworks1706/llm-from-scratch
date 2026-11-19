import dotenv from "dotenv";
import mongoose from "mongoose";
import connectDatabase from "./config/MongoDb.js";
import Product from "./Models/ProductModel.js";
import products from "./data/Products.js";

dotenv.config();

if (!process.env.MONGO_URL) {
  throw new Error("Set MONGO_URL to your demo database before seeding.");
}

try {
  await connectDatabase();
  const { upsertedCount } = await Product.bulkWrite(
    products.map((product) => ({
      updateOne: {
        filter: { name: product.name },
        update: { $setOnInsert: product },
        upsert: true,
      },
    })),
  );
  console.log(`Added ${upsertedCount} demo products.`);
} finally {
  await mongoose.disconnect();
}
