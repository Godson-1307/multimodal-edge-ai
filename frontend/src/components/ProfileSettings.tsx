import React, { useState } from 'react';
import { 
  Settings, 
  Moon, 
  Save, 
  LogOut, 
  Cpu, 
  CheckCircle2, 
  ShieldCheck,
  Sliders
} from 'lucide-react';
import { UserProfile, EdgeSettings } from '../types';

interface ProfileSettingsProps {
  user: UserProfile | null;
  onUpdateUser: (updated: UserProfile) => void;
  settings: EdgeSettings;
  onUpdateSettings: (newSettings: EdgeSettings) => void;
  onLogout: () => void;
}

export const ProfileSettings: React.FC<ProfileSettingsProps> = ({
  user,
  onUpdateUser,
  settings,
  onUpdateSettings,
  onLogout,
}) => {
  const [name, setName] = useState(user?.name || 'Sarah Connor');
  const [email, setEmail] = useState(user?.email || 'sarah.analyst@edgeai.dev');
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [profileSaved, setProfileSaved] = useState(false);

  // Settings state
  const [darkMode, setDarkMode] = useState(settings.darkMode);
  const [saveHistory, setSaveHistory] = useState(settings.saveHistory);
  const [modelType, setModelType] = useState(settings.modelType);
  const [deviceTarget, setDeviceTarget] = useState(settings.deviceTarget || 'edge-gpu');
  const [settingsSaved, setSettingsSaved] = useState(false);

  const handleSaveProfile = (e: React.FormEvent) => {
    e.preventDefault();
    onUpdateUser({
      name,
      email,
      role: user?.role || 'Edge AI Analyst',
    });
    setIsEditingProfile(false);
    setProfileSaved(true);
    setTimeout(() => setProfileSaved(false), 2500);
  };

  const handleSavePreferences = () => {
    onUpdateSettings({
      ...settings,
      darkMode,
      saveHistory,
      modelType,
      deviceTarget,
    });
    setSettingsSaved(true);
    setTimeout(() => setSettingsSaved(false), 2500);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-8">
      {/* Page Heading */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
          Profile &amp; Settings
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Manage your operator identity and application preferences.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-12 gap-8">
        
        {/* Profile Card (Left column) */}
        <div className="md:col-span-5 bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-6">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-blue-600 to-indigo-700 text-white flex items-center justify-center text-2xl font-bold shadow-md shadow-blue-500/20">
              {name.charAt(0).toUpperCase()}
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">{name}</h3>
              <p className="text-xs text-slate-500">{email}</p>
              <span className="inline-block mt-1 text-[10px] font-semibold uppercase px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200">
                {user?.role || 'Edge Operator'}
              </span>
            </div>
          </div>

          {profileSaved && (
            <div className="p-2.5 bg-emerald-50 text-emerald-700 text-xs rounded-lg border border-emerald-200 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>Profile updated successfully!</span>
            </div>
          )}

          {!isEditingProfile ? (
            <div className="space-y-4 pt-2 border-t border-slate-100 text-xs">
              <div>
                <span className="text-slate-400 block uppercase font-bold text-[10px]">Full Name</span>
                <span className="font-semibold text-slate-800">{name}</span>
              </div>
              <div>
                <span className="text-slate-400 block uppercase font-bold text-[10px]">Email Address</span>
                <span className="font-semibold text-slate-800">{email}</span>
              </div>
              <div>
                <span className="text-slate-400 block uppercase font-bold text-[10px]">System Access</span>
                <span className="font-semibold text-slate-800">Qwen2-VL Edge Inference Client</span>
              </div>

              <button
                id="edit-profile-btn"
                onClick={() => setIsEditingProfile(true)}
                className="w-full py-2.5 px-4 bg-slate-100 hover:bg-slate-200 text-slate-800 font-semibold rounded-xl transition-colors text-center block"
              >
                Edit Profile
              </button>
            </div>
          ) : (
            <form onSubmit={handleSaveProfile} className="space-y-4 pt-2 border-t border-slate-100 text-xs">
              <div>
                <label className="text-slate-700 font-bold block mb-1">Full Name</label>
                <input
                  type="text"
                  required
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full p-2 bg-slate-50 border border-slate-300 rounded-lg text-xs text-slate-900"
                />
              </div>
              <div>
                <label className="text-slate-700 font-bold block mb-1">Email Address</label>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full p-2 bg-slate-50 border border-slate-300 rounded-lg text-xs text-slate-900"
                />
              </div>

              <div className="flex gap-2">
                <button
                  type="submit"
                  className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg"
                >
                  Save Changes
                </button>
                <button
                  type="button"
                  onClick={() => setIsEditingProfile(false)}
                  className="px-3 py-2 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 font-medium"
                >
                  Cancel
                </button>
              </div>
            </form>
          )}

          <div className="pt-4 border-t border-slate-100">
            <button
              id="settings-logout-btn"
              onClick={onLogout}
              className="w-full py-2 px-4 rounded-xl text-xs font-semibold text-rose-600 hover:bg-rose-50 border border-rose-200 transition-colors flex items-center justify-center gap-1.5"
            >
              <LogOut className="w-3.5 h-3.5" />
              <span>Logout</span>
            </button>
          </div>
        </div>

        {/* Preferences Configuration (Right column) */}
        <div className="md:col-span-7 space-y-6">
          
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-5">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <Settings className="w-4 h-4 text-blue-600" />
                <span>Application Preferences</span>
              </h3>
              <span className="text-[11px] font-semibold text-slate-500 bg-slate-100 px-2 py-0.5 rounded">
                System Options
              </span>
            </div>

            {settingsSaved && (
              <div className="p-2.5 bg-emerald-50 text-emerald-700 text-xs rounded-lg border border-emerald-200 flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>Preferences saved successfully!</span>
              </div>
            )}

            <div className="space-y-4 text-xs">
              {/* Dark Mode toggle */}
              <label className="flex items-center justify-between p-3.5 rounded-xl bg-slate-50 cursor-pointer hover:bg-slate-100/80 transition-colors border border-slate-100">
                <div className="flex items-center gap-3">
                  <Moon className="w-4 h-4 text-slate-600" />
                  <div>
                    <span className="font-bold text-slate-800 block">Dark Mode</span>
                    <span className="text-slate-500 text-[11px]">Enable low-contrast high-density theme</span>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={darkMode}
                  onChange={(e) => setDarkMode(e.target.checked)}
                  className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500"
                />
              </label>

              {/* Save Processing History */}
              <label className="flex items-center justify-between p-3.5 rounded-xl bg-slate-50 cursor-pointer hover:bg-slate-100/80 transition-colors border border-slate-100">
                <div className="flex items-center gap-3">
                  <Save className="w-4 h-4 text-slate-600" />
                  <div>
                    <span className="font-bold text-slate-800 block">Save Processing History</span>
                    <span className="text-slate-500 text-[11px]">Keep local log of analyzed invoices and extractions</span>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={saveHistory}
                  onChange={(e) => setSaveHistory(e.target.checked)}
                  className="w-4 h-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500"
                />
              </label>

              {/* Model Selection */}
              <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100 space-y-2">
                <label className="font-bold text-slate-800 block flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-blue-600" />
                  <span>Default Model Architecture</span>
                </label>
                <select
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value as any)}
                  className="w-full p-2 bg-white border border-slate-200 rounded-lg text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="qwen2-vl-2b">Qwen2-VL-2B (QLoRA 4-bit Edge Optimized - Recommended)</option>
                  <option value="qwen2-vl-7b">Qwen2-VL-7B (High-Precision Full Attention)</option>
                </select>
                <p className="text-[11px] text-slate-500">
                  Select the neural model weights loaded for invoice extraction and OCR tokenization.
                </p>
              </div>

              {/* Hardware Acceleration Target */}
              <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-100 space-y-2">
                <label className="font-bold text-slate-800 block flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-indigo-600" />
                  <span>Hardware Inference Accelerator</span>
                </label>
                <select
                  value={deviceTarget}
                  onChange={(e) => setDeviceTarget(e.target.value as any)}
                  className="w-full p-2 bg-white border border-slate-200 rounded-lg text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="edge-gpu">Embedded GPU (CUDA / Vulkan / Metal Acceleration)</option>
                  <option value="npu">Dedicated NPU (Neural Processing Unit)</option>
                  <option value="cpu">Standard CPU (Multi-threaded AVX-512)</option>
                </select>
                <p className="text-[11px] text-slate-500">
                  Hardware backend utilized behind the scenes for vision-language inference.
                </p>
              </div>

              {/* Privacy Guarantee card */}
              <div className="p-3.5 rounded-xl bg-emerald-50/60 border border-emerald-200/70 flex items-start gap-3">
                <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-emerald-900 block">Privacy Guarantee</span>
                  <p className="text-[11px] text-emerald-700 leading-relaxed mt-0.5">
                    All document OCR processing runs securely in your local environment. Financial figures, vendor accounts, and line-item details remain private to your system.
                  </p>
                </div>
              </div>
            </div>

            <div className="pt-2">
              <button
                id="save-preferences-btn"
                onClick={handleSavePreferences}
                className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold text-xs rounded-xl shadow-sm transition-all"
              >
                Save Preferences
              </button>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
};
