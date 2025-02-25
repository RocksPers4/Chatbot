import React from 'react';

const AnimatedLogo = () => {
  return (
    <div className="flex items-center">
      <svg
        className="w-10 h-10 mr-2 animate-bounce"
        viewBox="0 0 100 100"
        fill="none"
        xmlns="https://www.canva.com/join/dkd-tyq-zxq"
      >
        <path
          d="M50 95C74.8528 95 95 74.8528 95 50C95 25.1472 74.8528 5 50 5C25.1472 5 5 25.1472 5 50C5 74.8528 25.1472 95 50 95Z"
          fill="#F4A261"
        />
        <path
          d="M35 40C38.3137 40 41 37.3137 41 34C41 30.6863 38.3137 28 35 28C31.6863 28 29 30.6863 29 34C29 37.3137 31.6863 40 35 40Z"
          fill="white"
        />
        <path
          d="M65 40C68.3137 40 71 37.3137 71 34C71 30.6863 68.3137 28 65 28C61.6863 28 59 30.6863 59 34C59 37.3137 61.6863 40 65 40Z"
          fill="white"
        />
        <path
          d="M35 37C36.6569 37 38 35.6569 38 34C38 32.3431 36.6569 31 35 31C33.3431 31 32 32.3431 32 34C32 35.6569 33.3431 37 35 37Z"
          fill="black"
        />
        <path
          d="M65 37C66.6569 37 68 35.6569 68 34C68 32.3431 66.6569 31 65 31C63.3431 31 62 32.3431 62 34C62 35.6569 63.3431 37 65 37Z"
          fill="black"
        />
        <path
          d="M50 70C59.9411 70 68 61.9411 68 52H32C32 61.9411 40.0589 70 50 70Z"
          fill="white"
        />
      </svg>
      <span className="text-2xl font-bold">PochiBot</span>
    </div>
  );
};

export default AnimatedLogo;
