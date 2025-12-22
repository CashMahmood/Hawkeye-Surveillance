import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, History, Settings, ShieldAlert, Radio } from 'lucide-react';
import { motion } from 'framer-motion';
import Dashboard from './pages/Dashboard';
import Events from './pages/Events';

const Sidebar: React.FC = () => {
  const location = useLocation();
  const [backendStatus, setBackendStatus] = useState<'ONLINE' | 'OFFLINE'>('OFFLINE');

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch('http://localhost:8000/health');
        if (res.ok) setBackendStatus('ONLINE');
        else setBackendStatus('OFFLINE');
      } catch {
        setBackendStatus('OFFLINE');
      }
    };
    checkStatus();
    const timer = setInterval(checkStatus, 5000);
    return () => clearInterval(timer);
  }, []);

  const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/events', icon: History, label: 'Hist. Archive' },
  ];

  return (
    <div className="w-20 flex flex-col items-center py-8 gap-10 border-r border-tactical-green/20 bg-black/40 backdrop-blur-xl z-50">
      <Link to="/" className="relative group">
        <div className="p-3 border-2 border-tactical-green/40 rounded-sm bg-tactical-green/5 hover:bg-tactical-green/10 transition-all shadow-[0_0_15px_rgba(0,255,65,0.1)] active:scale-90">
          <ShieldAlert className="w-8 h-8 neon-text-green" />
        </div>
        <div className="absolute -top-1 -right-1 w-3 h-3 bg-tactical-green rounded-full border-2 border-background animate-pulse" />
      </Link>

      <nav className="flex flex-col gap-8 flex-1">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`p-4 rounded-sm transition-all relative group flex items-center justify-center ${isActive
                ? 'bg-tactical-green/10 text-tactical-green shadow-glow border border-tactical-green/30'
                : 'text-tactical-green/30 hover:text-tactical-green hover:bg-tactical-green/5'
                }`}
            >
              <item.icon className={`w-6 h-6 ${isActive ? 'drop-shadow-[0_0_8px_#00ff41]' : ''}`} />

              {/* Tooltip */}
              <div className="absolute left-full ml-6 px-3 py-1 bg-black border border-tactical-green/40 text-[10px] uppercase font-black tracking-[0.2em] invisible group-hover:visible whitespace-nowrap z-50 shadow-2xl skew-x-[-10deg]">
                {item.label}
              </div>

              {isActive && (
                <motion.div
                  layoutId="activeTab"
                  className="absolute -left-1 w-1.5 h-full bg-tactical-green shadow-glow rounded-r-full"
                />
              )}
            </Link>
          );
        })}
      </nav>

      <div className="flex flex-col items-center gap-6 mt-auto">
        <div className="flex flex-col items-center group cursor-help">
          <Radio className={`w-5 h-5 ${backendStatus === 'ONLINE' ? 'text-tactical-green animate-pulse' : 'text-tactical-red'}`} />
          <span className="text-[8px] font-black mt-1 opacity-40">{backendStatus}</span>
        </div>
        <div className="p-3 text-tactical-green/30 hover:text-tactical-green transition-colors cursor-pointer active:scale-90">
          <Settings className="w-6 h-6 border border-transparent hover:border-tactical-green/20 rounded-sm p-1" />
        </div>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <Router>
      <div className="flex h-screen bg-background text-tactical-green font-mono overflow-hidden">
        {/* Global HUD Overlay Lines */}
        <div className="fixed inset-0 pointer-events-none border-[12px] border-tactical-green/5 z-[60]" />

        <Sidebar />

        <main className="flex-1 overflow-hidden relative">
          {/* Subtle Ambient HUD elements */}
          <div className="absolute top-6 right-8 text-[9px] font-black opacity-20 tracking-widest z-0 pointer-events-none">
            HAWKEYE_SYSTEM_DOM_READY // OPERATIONAL
          </div>

          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/events" element={<Events />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;
