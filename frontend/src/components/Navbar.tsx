import React from 'react';
import { AppView, UserProfile } from '../types';
import { 
  FileText, 
  LayoutDashboard, 
  UploadCloud, 
  History, 
  User, 
  Cpu, 
  LogOut, 
  LogIn,
  CheckCircle2
} from 'lucide-react';

interface NavbarProps {
  currentView: AppView;
  onNavigate: (view: AppView) => void;
  user: UserProfile | null;
  onLogout: () => void;
  onLoginClick: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentView,
  onNavigate,
  user,
  onLogout,
  onLoginClick,
}) => {
  return (
    <header className="sticky top-0 z-40 bg-white/95 backdrop-blur-md border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Brand */}
          <div className="flex items-center gap-3">
            <button
              id="brand-logo-btn"
              onClick={() => onNavigate(user ? 'dashboard' : 'landing')}
              className="flex items-center gap-2.5 text-left group focus:outline-none"
            >
              <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 via-blue-600 to-indigo-700 flex items-center justify-center text-white shadow-sm shadow-blue-500/20 group-hover:scale-105 transition-transform">
                <FileText className="w-5 h-5 text-white" />
              </div>
              <div>
                <span className="font-bold text-slate-900 tracking-tight text-base sm:text-lg flex items-center gap-1.5">
                  Multi-Modal Edge AI
                  <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200/60 hidden sm:inline-flex items-center gap-1">
                    <Cpu className="w-3 h-3 text-blue-600" />
                    Qwen2-VL
                  </span>
                </span>
                <p className="text-xs text-slate-500 hidden md:block">Document OCR & Invoice Extraction</p>
              </div>
            </button>
          </div>

          {/* Center Navigation Links */}
          <nav className="hidden md:flex items-center gap-1">
            {user ? (
              <>
                <button
                  id="nav-dashboard-btn"
                  onClick={() => onNavigate('dashboard')}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                    currentView === 'dashboard'
                      ? 'bg-slate-100 text-blue-700 font-semibold'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                  }`}
                >
                  <LayoutDashboard className="w-4 h-4" />
                  Dashboard
                </button>

                <button
                  id="nav-upload-btn"
                  onClick={() => onNavigate('upload')}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                    currentView === 'upload' || currentView === 'processing'
                      ? 'bg-slate-100 text-blue-700 font-semibold'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                  }`}
                >
                  <UploadCloud className="w-4 h-4" />
                  Upload Document
                </button>

                <button
                  id="nav-history-btn"
                  onClick={() => onNavigate('history')}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                    currentView === 'history' || currentView === 'details'
                      ? 'bg-slate-100 text-blue-700 font-semibold'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                  }`}
                >
                  <History className="w-4 h-4" />
                  History
                </button>
              </>
            ) : (
              <button
                id="nav-home-btn"
                onClick={() => onNavigate('landing')}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  currentView === 'landing'
                    ? 'bg-slate-100 text-blue-700 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                Home
              </button>
            )}
          </nav>

          {/* Right Action & User Profile */}
          <div className="flex items-center gap-3">
            {/* Edge status badge */}
            <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200/80 text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>Edge AI: Local Ready</span>
            </div>

            {user ? (
              <div className="flex items-center gap-2">
                <button
                  id="nav-profile-btn"
                  onClick={() => onNavigate('profile')}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors border ${
                    currentView === 'profile'
                      ? 'border-blue-300 bg-blue-50/50 text-blue-800'
                      : 'border-slate-200 bg-white hover:bg-slate-50 text-slate-700'
                  }`}
                >
                  <div className="w-6 h-6 rounded-full bg-blue-600 text-white flex items-center justify-center text-xs font-bold">
                    {user.name.charAt(0).toUpperCase()}
                  </div>
                  <span className="font-medium hidden sm:inline">{user.name.split(' ')[0]}</span>
                </button>

                <button
                  id="nav-logout-btn"
                  onClick={onLogout}
                  title="Logout"
                  className="p-2 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-lg transition-colors"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <button
                  id="nav-login-btn"
                  onClick={onLoginClick}
                  className="px-3.5 py-2 text-sm font-medium text-slate-700 hover:text-blue-700 transition-colors"
                >
                  Login
                </button>
                <button
                  id="nav-get-started-btn"
                  onClick={() => onNavigate('register')}
                  className="px-4 py-2 text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-all shadow-blue-500/20 hover:shadow-blue-500/30"
                >
                  Get Started
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Mobile Navigation bar */}
        <div className="flex md:hidden border-t border-slate-100 py-2 items-center justify-around text-xs">
          {user ? (
            <>
              <button
                onClick={() => onNavigate('dashboard')}
                className={`flex flex-col items-center gap-1 ${
                  currentView === 'dashboard' ? 'text-blue-600 font-semibold' : 'text-slate-500'
                }`}
              >
                <LayoutDashboard className="w-4 h-4" />
                <span>Dashboard</span>
              </button>
              <button
                onClick={() => onNavigate('upload')}
                className={`flex flex-col items-center gap-1 ${
                  currentView === 'upload' ? 'text-blue-600 font-semibold' : 'text-slate-500'
                }`}
              >
                <UploadCloud className="w-4 h-4" />
                <span>Upload</span>
              </button>
              <button
                onClick={() => onNavigate('history')}
                className={`flex flex-col items-center gap-1 ${
                  currentView === 'history' ? 'text-blue-600 font-semibold' : 'text-slate-500'
                }`}
              >
                <History className="w-4 h-4" />
                <span>History</span>
              </button>
            </>
          ) : (
            <button
              onClick={() => onNavigate('landing')}
              className={`flex flex-col items-center gap-1 ${
                currentView === 'landing' ? 'text-blue-600 font-semibold' : 'text-slate-500'
              }`}
            >
              <FileText className="w-4 h-4" />
              <span>Home</span>
            </button>
          )}
          {user && (
            <button
              onClick={() => onNavigate('profile')}
              className={`flex flex-col items-center gap-1 ${
                currentView === 'profile' ? 'text-blue-600 font-semibold' : 'text-slate-500'
              }`}
            >
              <User className="w-4 h-4" />
              <span>Profile</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
};
