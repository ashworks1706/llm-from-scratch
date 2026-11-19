import React from "react";

const Contact = () => {
  window.scrollTo(0, 0);

  return (
    <section id="contact">
      <div className="contact-container">
        <h1 className="aight font-serif font-bold">Contact</h1>
        <div className="social-media">
          <h3>
            Let's have a talk! My Socials are always open, Leave a message.
          </h3>
          <div className="socials">
            <a href="https://github.com/VEcTorX009">
              <i className="fa-brands fa-github"></i>
            </a>
            <a href="https://www.instagram.com/som.developer/">
              <i className="fa-brands fa-instagram"></i>
            </a>
            <a href="https://www.linkedin.com/in/somsrivastava">
              <i className="fa-brands fa-linkedin"></i>
            </a>
            <a href="https://www.discordapp.com/users/897822597377581086/">
              <i className="fa-brands fa-discord"></i>
            </a>
            <a href="https://twitter.com/home">
              <i className="fa-brands fa-twitter"></i>
            </a>
          </div>
        </div>

        <form
          action="https://formsubmit.co/e4eca9a791152521a6c297a1bb829df5"
          method="POST"
          className="reveal3"
        >
          <h5>get in touch.</h5>

          <h3>Name</h3>
          <input type="text" name="Name" id="name" required />
          <h3>Email</h3>
          <input type="email" name="Email" id="email" required />
          <h3>Message</h3>
          <textarea
            id="message"
            name="Message"
            rows="4"
            maxlength="250"
          ></textarea>
          <button type="submit">Send</button>
        </form>
      </div>
    </section>
  );
};

export default Contact;
