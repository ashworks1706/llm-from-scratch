import React from "react"
import Header from "../components/Header"
import Main from "../compoentns/Home/Main"
import Sidebar from "../components/Sidebar"

const HomeScreen = ()=>{
    return (
        <>
            <Sidebar/>
            <main className="main-wrap">
                <Header/>
                <Main/>
            </main>
        </>
    )
}
export default HomeScreen