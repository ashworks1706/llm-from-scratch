import { React, {useEffect} } from "react-dom";
import {Link} from "react-router-dom"
import $ from "jquery"

const Header =() =>{
    useEffect(() => {
      $("[data-trigger]").on("click", function (e) {
        e.preventDefault()
        e.stopPropagation()
        var offcanvas_id = $(this).attr("data-trigger");
        $(offcanvas_id).toggleClass("show")
      })
      $(".btn-aside-minimize").on("click",function () {
        if(window.innerWidth < 768)
        {
            $("body").removeClass("aside-mini")
            $(".navbar-aside").removeClass("show")
        }
        else{
            $("body").toggleClass("aside-mini")
        }
      })
    }, [])
    
}