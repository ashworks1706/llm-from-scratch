import React from 'react'
import {Redirect, Route} from "react-router-dom"
export default function PrivateRouter({component: Component, ...rest}) {
  return (
    <Route {...rest} component={(props)=>{
        const token = window.localStorage.getItem("useInfo")
        if (!token) {
            return <Component {...props}/>
        }
        else{
            return <Redirect to="/login" />
        }
    }}
    />
  )
}
