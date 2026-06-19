import { useState } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  NavLink,
} from "react-router-dom";

import Home from "./pages/Home";
import FallDashboard from "./pages/FallDashboard";
// import AbnormalDashboard from "./pages/AbnormalDashboard";

import "./App.css";

function App() {
  const [menuOpen, setMenuOpen] = useState(false);

  const closeMenu = () => {
    setMenuOpen(false);
  };

  return (
    <BrowserRouter>
      <div className="app-layout">
        {/* 햄버거 버튼 */}
        <button className="menu-button" onClick={() => setMenuOpen(true)}>
          ☰
        </button>

        {/* 어두운 배경 */}
        {menuOpen && <div className="menu-backdrop" onClick={closeMenu}></div>}

        {/* 사이드 메뉴 */}
        <aside className={`side-menu ${menuOpen ? "open" : ""}`}>
          <div className="side-menu-header">
            <div>
              <strong>Smart Care AI</strong>
              <span>협업 관제 플랫폼</span>
            </div>

            <button onClick={closeMenu}>×</button>
          </div>

          <nav className="side-nav">
            <NavLink to="/" onClick={closeMenu}>
              홈
            </NavLink>

            <NavLink to="/fall" onClick={closeMenu}>
              낙상 관제
            </NavLink>

            <NavLink to="/abnormal" onClick={closeMenu}>
              이상행동 분석
            </NavLink>
          </nav>
        </aside>

        <main className="app-main">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/fall" element={<FallDashboard />} />
            {/* <Route path="/abnormal" element={<AbnormalDashboard />} /> */}

            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
