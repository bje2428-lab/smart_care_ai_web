import React from "react";
import Home from "./pages/Home.jsx";
import FallDashboard from "./pages/FallDashboard.jsx";

function App() {
  const [page, setPage] = React.useState("home");
  const [menuOpen, setMenuOpen] = React.useState(false);

  const movePage = (target) => {
    setPage(target);
    setMenuOpen(false);
  };

  return (
    <>
      <div className="site-topbar">
        <div className="site-brand" onClick={() => movePage("home")}>
          <div className="brand-mark">S</div>
          <div>
            <strong>Smart Care AI</strong>
            <span>독거노인 안전 관리 협업 관제 플랫폼</span>
          </div>
        </div>

        <button className="hamburger-menu" onClick={() => setMenuOpen(true)}>
          <span></span>
          <span></span>
          <span></span>
        </button>
      </div>

      {menuOpen && (
        <div className="menu-backdrop" onClick={() => setMenuOpen(false)}>
          <nav className="drawer-menu" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-top">
              <div>
                <h2>MENU</h2>
                <p>Smart Care AI</p>
              </div>
              <button onClick={() => setMenuOpen(false)}>×</button>
            </div>

            <button
              className={page === "home" ? "active" : ""}
              onClick={() => movePage("home")}
            >
              🏠 프로젝트 소개
            </button>

            <button
              className={page === "fall" ? "active" : ""}
              onClick={() => movePage("fall")}
            >
              🚨 노인낙상 관제시스템
            </button>
          </nav>
        </div>
      )}

      {page === "home" && <Home movePage={movePage} />}
      {page === "fall" && <FallDashboard />}
    </>
  );
}

export default App;
