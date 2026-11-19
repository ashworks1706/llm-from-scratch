import mongoose from 'mongoose';
mongoose.set("strictQuery", true);
const connectDatabase = async() =>{
        return mongoose.connect(process.env.MONGO_URL,{
            useNewUrlParser : true,
            useUnifiedTopology: true,
        }).then(() => {
            console.log("connected to mongodb")
          })
}
export default connectDatabase