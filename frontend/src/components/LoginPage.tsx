import React, { useState } from 'react';
import { FileText, Lock, Mail, ArrowRight, Sparkles, CheckCircle2 } from 'lucide-react';
import { AppView, UserProfile } from '../types';

interface LoginPageProps {
  onLoginSuccess: (user: UserProfile) => void;
  onNavigate: (view: AppView) => void;
}

export const LoginPage: React.FC<LoginPageProps> = ({ onLoginSuccess, onNavigate }) => {
  const [email, setEmail] = useState('sarah.analyst@edgeai.dev');
  const [password, setPassword] = useState('••••••••••••');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Please enter your email and password');
      return;
    }
    // Mock authentication
    onLoginSuccess({
      name: email.split('@')[0].replace('.', ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
      email,
      role: 'Edge AI Research Analyst',
    });
  };

  const handleDemoLogin = (roleName: string, demoEmail: string) => {
    setEmail(demoEmail);
    setPassword('demopass123');
    onLoginSuccess({
      name: roleName,
      email: demoEmail,
      role: 'Project Evaluator / Viva Demo',
    });
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] flex items-center justify-center p-4 sm:p-6 bg-slate-50">
      <div className="w-full max-w-md bg-white rounded-2xl border border-slate-200 shadow-xl shadow-slate-200/50 p-6 sm:p-8 space-y-6">
        {/* Brand header */}
        <div className="text-center space-y-2">
          <div className="w-12 h-12 rounded-xl bg-blue-600 text-white flex items-center justify-center mx-auto shadow-md shadow-blue-500/20">
            <FileText className="w-6 h-6" />
          </div>
          <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Login to Edge AI</h2>
          <p className="text-xs text-slate-500">Document OCR &amp; Qwen2-VL Extraction Workspace</p>
        </div>

        {error && (
          <div className="p-3 text-xs bg-rose-50 text-rose-700 border border-rose-200 rounded-lg">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5">
              Email Address
            </label>
            <div className="relative">
              <Mail className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
              <input
                id="login-email-input"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@organization.com"
                className="w-full pl-9 pr-4 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white text-slate-900"
              />
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider">
                Password
              </label>
              <button
                type="button"
                onClick={() => alert('Demo password reset link has been simulated.')}
                className="text-xs text-blue-600 hover:text-blue-800 font-medium"
              >
                Forgot Password?
              </button>
            </div>
            <div className="relative">
              <Lock className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
              <input
                id="login-password-input"
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                className="w-full pl-9 pr-4 py-2.5 bg-slate-50 border border-slate-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white text-slate-900"
              />
            </div>
          </div>

          <button
            id="login-submit-btn"
            type="submit"
            className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-sm rounded-xl shadow-md shadow-blue-500/20 hover:shadow-lg transition-all flex items-center justify-center gap-2"
          >
            <span>Login</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </form>

        {/* Demo Fast Login */}
        <div className="pt-2 border-t border-slate-100">
          <p className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2 text-center">
            College Project Quick Demo Access
          </p>
          <button
            id="demo-login-btn"
            type="button"
            onClick={() => handleDemoLogin('Sarah Connor', 'sarah.analyst@edgeai.dev')}
            className="w-full py-2 px-3 bg-slate-100 hover:bg-slate-200 text-slate-800 text-xs font-medium rounded-lg transition-colors flex items-center justify-center gap-1.5"
          >
            <Sparkles className="w-3.5 h-3.5 text-blue-600" />
            <span>Sign in as Sarah (Demo Account)</span>
          </button>
        </div>

        {/* Create account toggle */}
        <div className="text-center text-xs text-slate-500">
          Don't have an account?{' '}
          <button
            id="switch-to-register-btn"
            type="button"
            onClick={() => onNavigate('register')}
            className="text-blue-600 hover:text-blue-800 font-semibold"
          >
            Create Account
          </button>
        </div>
      </div>
    </div>
  );
};
