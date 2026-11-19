import './App.css';
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
import Header from "./components/Main/Header"
import Navbar from "./components/Main/Navbar"
import Footer from "./components/Main/Footer"
import "react-toastify/dist/ReactToastify.css"
import Shop from "./components/Main/Shop"
import Login from "./screens/Login"
import RegisterPageScreen from "./screens/RegisterPage"
import CartScreen from "./screens/CartScreen"
import PaymentScreen from "./screens/PaymentScreen"
import PlaceOrder from './screens/PlaceOrder';
import ProfileScreen from './components/profileComponents/ProfileScreen';
import Orders from './components/profileComponents/Orders';
import ShippingScreen from './screens/ShippingScreen';
import OrderScreen from './screens/OrderScreen';
import ProductDetailsPage from './screens/ProductDetailsPage';
import Contact from './components/Main/Contact';
import Error from './components/LoadingError/Error';
import OfflineError  from "./screens/Offlines";
import Redirect from './components/Main/Redirect';
import { Offline, Online } from "react-detect-offline";
import PrivateRouter from './PrivateRouter';
import Search from './screens/Search';

function App() {
  return (
    <div className="p-0 m-0">
    <Online>
      <Router>
      <Navbar/>
      <Switch>
      <Route path={"/"} component={Header} exact/>
      <Route path={"/register"} component={RegisterPageScreen} />
      <Route path={"/look"}   component={Search} />
      <Route path={"/login"} component={Login} />      
      <PrivateRouter path={"/profile"} component={ProfileScreen} />      
      <Route path={"/shop"}  component={Shop} />
      <Route path={"/search/:keyword"}  component={Shop} />
      <Route path={"/products/:id"} component={ProductDetailsPage} />
      <Route path={"/cart/:id?"} component={CartScreen} />
      <Route path={"/order/:id"} component={OrderScreen} />
      <Route path={"/shipping"} component={ShippingScreen} />
      <Route path={"/placeorder"} component={PlaceOrder} />
      <Route path={"/payment"} component={PaymentScreen} />
      <Route path={"/contact"} component={Contact} />
      <Route path={"/about"} component={Redirect} />
      <Route path={"/orders"} component={Orders} />
      <Route path={"*"} component={Error} />
      </Switch>
      <Footer/>
      </Router>
    </Online>
    <Offline>
      <OfflineError/>
    </Offline>

    </div>
  );
}

export default App;
