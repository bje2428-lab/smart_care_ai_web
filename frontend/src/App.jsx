import { useState } from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  NavLink,
  useLocation,
} from "react-router-dom";

import Home from "./pages/Home";
import FallDashboard from "./pages/FallDashboard";
import AbnormalDashboard from "./pages/AbnormalDashboard";
import VitalDashboard from "./pages/VitalDashboard";
import IntegratedDashboard from "./pages/IntegratedDashboard";

import "./App.css";

function App() {
  const [menuOpen, setMenuOpen] = useState(false);

  const closeMenu = () => {
    setMenuOpen(false);
  };

  return (
    <BrowserRouter>
      <div className="app-layout">
        <button className="menu-button" onClick={() => setMenuOpen(true)}>
          ☰
        </button>

        {menuOpen && <div className="menu-backdrop" onClick={closeMenu}></div>}

        <SideMenu menuOpen={menuOpen} closeMenu={closeMenu} />

        <main className="app-main">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/integrated" element={<IntegratedDashboard />} />
            <Route path="/fall" element={<FallDashboard />} />
            <Route path="/abnormal" element={<AbnormalDashboard />} />
            <Route path="/vital" element={<VitalDashboard />} />

            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

function SideMenu({ menuOpen, closeMenu }) {
  const location = useLocation();
  const [analysisOpen, setAnalysisOpen] = useState(false);

  const isAnalysisPage =
    location.pathname === "/fall" ||
    location.pathname === "/abnormal" ||
    location.pathname === "/vital";

  const showAnalysisMenu = analysisOpen || isAnalysisPage;

  const toggleAnalysisMenu = () => {
    setAnalysisOpen((prev) => !prev);
  };

  return (
    <aside className={`side-menu ${menuOpen ? "open" : ""}`}>
      <div className="side-menu-header">
        <div>
          <strong>Smart Care AI</strong>
          <span>AI 통합 관제 플랫폼</span>
        </div>

        <button onClick={closeMenu}>×</button>
      </div>

      <nav className="side-nav">
        <NavLink
          to="/"
          end
          onClick={closeMenu}
          className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
        >
          홈
        </NavLink>

        <NavLink
          to="/integrated"
          onClick={closeMenu}
          className={({ isActive }) =>
            `nav-link integrated-link ${isActive ? "active" : ""}`
          }
        >
          <span className="integrated-dot"></span>
          통합관제
        </NavLink>

        <div
          className={`analysis-menu-wrap ${
            showAnalysisMenu ? "open" : ""
          } ${isAnalysisPage ? "active-group" : ""}`}
          onMouseEnter={() => setAnalysisOpen(true)}
          onMouseLeave={() => {
            if (!isAnalysisPage) {
              setAnalysisOpen(false);
            }
          }}
        >
          <button
            type="button"
            className="analysis-menu-button"
            onClick={toggleAnalysisMenu}
          >
            <span>개별 분석 메뉴</span>
            <span className="analysis-arrow">
              {showAnalysisMenu ? "⌃" : "⌄"}
            </span>
          </button>

          <div className="analysis-sub-menu">
            <NavLink
              to="/fall"
              onClick={closeMenu}
              className={({ isActive }) =>
                `analysis-sub-link ${isActive ? "active" : ""}`
              }
            >
              낙상 관제
            </NavLink>

            <NavLink
              to="/abnormal"
              onClick={closeMenu}
              className={({ isActive }) =>
                `analysis-sub-link ${isActive ? "active" : ""}`
              }
            >
              이상행동 분석
            </NavLink>
          </div>
        </div>
      </nav>
    </aside>
  );
}

export default App;
