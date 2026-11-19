import React from 'react'
import { Link, NavLink } from "react-router-dom";

const Sidebar = () => {
  return (
    <div>
        <aside className="navbar-aside" id="offcanvas_aside">
            <div className="aside-top">
                <Link to="/" className="brand-wrap">
                    <img src="../../public/logo.jpg" alt="Books of Gold" />
                </Link>
                <div>
                    <button>
                        <i className="text-muted fas fa-stream"></i>
                    </button>
                </div>
            </div>

            <nav>
                <ul>
                    <li>
                        <NavLink activeClassName ="active" to="/" exact={true}>
                            <i className="icon fas fa-home"></i>
                            <span>Dashboard</span>
                        </NavLink>
                    </li>
                    <li className="menu-item">
                        <NavLink activeClassName="active" className="menu-link" to="/products">
                            <i className="icon fas fa-shopping-bag"></i>
                            <span>Books</span>
                        </NavLink>
                    </li>
                    <li className="menu-item">
                        <NavLink activeClassName="active" className="menu-link" to="/addproduct">
                            <i className="icon fas fa-cart-plus"></i>
                            <span>Add Book</span>
                        </NavLink>
                    </li>
                    <li className="menu-item">
                        <NavLink activeClassName="active" className="menu-link" to="/category">
                            <i className="icon fas fa-list"></i>
                            <span>Category</span>
                        </NavLink>
                    </li>
                    <li className="menu-item">
                        <NavLink activeClassName="active" className="menu-link" to="/orders">
                            <i className="icon fas fa-bags-shopping"></i>
                            <span>Orders</span>
                        </NavLink>
                    </li>
                    <li className="menu-item">
                        <NavLink activeClassName="active" className="menu-link" to="/users">
                            <i className="icon fas fa-user"></i>
                            <span>Users</span>
                        </NavLink>
                    </li>
                    <li className="menu-item">
                        <NavLink activeClassName="active" className="menu-link" to="/sellers">
                            <i className="icon fas fa-store-alt"></i>
                            <span>Sellers</span>
                        </NavLink>
                    </li>
                    <li className="menu-item">
                        <NavLink activeClassName="active" className="menu-link" to="/transaction">
                            <i className="icon fas fa-usd-circle"></i>
                            <span>Transaction</span>
                        </NavLink>
                    </li>
                </ul>
            </nav>
        </aside>
    </div>
  )
}

export default Sidebar