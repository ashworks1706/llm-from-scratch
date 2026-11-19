import React from "react";
import { ReactNavbar } from "overlay-navbar";
import logo from "../../data/img/logo.jpg";
import { MdAccountCircle } from "react-icons/md";
import { MdSearch } from "react-icons/md";
import { MdAddShoppingCart } from "react-icons/md";
const Header = () => {
  return (


      <ReactNavbar 
        navColor1= "wheat"
        burgerColor="#b0a200"
        burgerColorHover="#dfd23c"
        logo={logo}
        logoWidth="15vmax"
        logoHoverSize="10px"
        logoHoverColor="#b0a200"
        link1Text="Home"
        link2Text="Shop"
        link3Text="Contact"
        link4Text="About"
        link1Url="/"
        link2Url="/shop"
        link3Url="/contact"
        link4Url="/about"
        cartIconUrl="/cart"
        profileIconUrl="/profile"
        link1Size="1.2vmax"
        link1Color="rgba(35, 35, 35,0.8)"
        nav1justifyContent="flex-end"
        nav2justifyContent="flex-end"
        nav3justifyContent="flex-start"
        nav4justifyContent="flex-start"
        link1ColorHover="#b0a200"
        link1Margin="1vmax"
        profileIconColor="black"
        searchIconColor="black"
        cartIconColor="rgba(35, 35, 35,0.8)"
        profileIconColorHover="#b0a200"
        searchIconColorHover="#b0a200"
        cartIconColorHover="#b0a200"
        cartIconMargin="1vmax"
        profileIcon={true}
        ProfileIconElement={MdAccountCircle}
        searchIcon={true}
        searchIconUrl="/look"
        SearchIconElement={MdSearch}
        cartIcon={true}
        CartIconElement={MdAddShoppingCart}
      />

  );
};

export default Header;