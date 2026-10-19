/**
 * Multi-Modal Edge AI — Document OCR & Invoice Extraction
 * Complete Frontend Application
 */

import React, { useState, useEffect } from 'react';

import {
  AppView,
  ExtractionResult,
  UserProfile,
  EdgeSettings,
} from './types';

import {
  INITIAL_HISTORY,
  MOCK_INVOICE_TOPSTAR,
} from './mockData';

import { analyzeDocument } from './services/api';

import { Navbar } from './components/Navbar';
import { LandingPage } from './components/LandingPage';
import { LoginPage } from './components/LoginPage';
import { RegisterPage } from './components/RegisterPage';
import { Dashboard } from './components/Dashboard';
import { UploadDocument } from './components/UploadDocument';
import { ProcessingView } from './components/ProcessingView';
import { ExtractionResults } from './components/ExtractionResults';
import { DocumentHistory } from './components/DocumentHistory';
import { DocumentDetails } from './components/DocumentDetails';
import { ProfileSettings } from './components/ProfileSettings';

export default function App() {
  // --------------------------------------------------
  // Current user
  // --------------------------------------------------

  const [user, setUser] = useState<UserProfile | null>({
    name: 'Sarah Connor',
    email: 'sarah.analyst@edgeai.dev',
    role: 'Edge AI Research Analyst',
  });

  // --------------------------------------------------
  // Current application view
  // --------------------------------------------------

  const [currentView, setCurrentView] =
    useState<AppView>('dashboard');

  // --------------------------------------------------
  // Document history
  // --------------------------------------------------

  const [history, setHistory] = useState<ExtractionResult[]>(() => {
    const saved = localStorage.getItem('edge_ocr_history');

    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (error) {
        console.warn('Unable to load saved history.');
        return INITIAL_HISTORY;
      }
    }

    return INITIAL_HISTORY;
  });

  // --------------------------------------------------
  // Application settings
  // --------------------------------------------------

  const [settings, setSettings] = useState<EdgeSettings>({
    darkMode: false,
    saveHistory: true,

    // FastAPI backend
    backendApiUrl: 'http://localhost:8000/api/analyze',

    useCustomBackend: false,

    // Current project model
    modelType: 'qwen2-vl-2b',

    // Local GPU target
    deviceTarget: 'edge-gpu',

    // Used by the frontend processing UI if required
    simulateDelay: 500,
  });

  // --------------------------------------------------
  // Currently displayed extraction result
  // --------------------------------------------------

  const [activeResult, setActiveResult] =
    useState<ExtractionResult>(MOCK_INVOICE_TOPSTAR);

  // --------------------------------------------------
  // Selected history document
  // --------------------------------------------------

  const [selectedHistoryDoc, setSelectedHistoryDoc] =
    useState<ExtractionResult | null>(null);

  // --------------------------------------------------
  // Processing state
  // --------------------------------------------------

  const [processingStage, setProcessingStage] = useState(0);

  const [processingFileName, setProcessingFileName] =
    useState('');

  // --------------------------------------------------
  // Persist history
  // --------------------------------------------------

  useEffect(() => {
    if (!settings.saveHistory) {
      return;
    }

    try {
      localStorage.setItem(
        'edge_ocr_history',
        JSON.stringify(history)
      );
    } catch (error) {
      console.warn(
        'Storage quota exceeded or localStorage unavailable.'
      );
    }
  }, [history, settings.saveHistory]);

  // --------------------------------------------------
  // Navigation
  // --------------------------------------------------

  const handleNavigate = (view: AppView) => {
    setCurrentView(view);

    window.scrollTo({
      top: 0,
      behavior: 'smooth',
    });
  };

  // --------------------------------------------------
  // Start document analysis
  // --------------------------------------------------

  const handleStartAnalysis = async (
    file:
      | File
      | {
          name: string;
          size: number;
          type: string;
          dataUrl?: string;
        }
  ) => {
    setProcessingFileName(file.name);
    setProcessingStage(0);
    setCurrentView('processing');

    try {
      const result = await analyzeDocument(file, {
        apiUrl: settings.backendApiUrl,

        simulateDelay: settings.simulateDelay,

        onStageChange: (stageIndex) => {
          setProcessingStage(stageIndex);
        },
      });

      // Store the real extraction result
      setActiveResult(result);

      // Save to history if enabled
      if (settings.saveHistory) {
        setHistory((previousHistory) => [
          result,
          ...previousHistory,
        ]);
      }

      // Show extraction results
      setCurrentView('results');
    } catch (error: any) {
      console.error('Inference failed:', error);

      const message =
        error?.message ||
        'Unable to process the document.';

      alert(`Inference failed: ${message}`);

      setCurrentView('upload');
    }
  };

  // --------------------------------------------------
  // Re-analyze an existing document
  // --------------------------------------------------

  const handleAnalyzeAgain = async (
    doc: ExtractionResult
  ) => {
    try {
      /*
       * History stores the browser preview URL.
       *
       * If the URL is still available, fetch it and
       * convert the resulting Blob back into a File.
       */

      if (!doc.fileUrl) {
        throw new Error(
          'The original document is no longer available.'
        );
      }

      const response = await fetch(doc.fileUrl);

      if (!response.ok) {
        throw new Error(
          'Unable to retrieve the original document.'
        );
      }

      const blob = await response.blob();

      const file = new File(
        [blob],
        doc.fileName,
        {
          type:
            doc.fileType ||
            blob.type ||
            'image/jpeg',
        }
      );

      await handleStartAnalysis(file);
    } catch (error: any) {
      console.error(
        'Re-analysis failed:',
        error
      );

      alert(
        `Unable to re-analyze this document: ${
          error?.message ||
          'Original file is unavailable.'
        }`
      );
    }
  };

  // --------------------------------------------------
  // Select a document from history
  // --------------------------------------------------

  const handleSelectDocument = (
    doc: ExtractionResult
  ) => {
    setSelectedHistoryDoc(doc);

    setActiveResult(doc);

    setCurrentView('details');
  };

  // --------------------------------------------------
  // Logout
  // --------------------------------------------------

  const handleLogout = () => {
    setUser(null);

    setCurrentView('landing');
  };

  // --------------------------------------------------
  // Render application
  // --------------------------------------------------

  return (
    <div
      className={`
        min-h-screen
        flex
        flex-col
        font-['Plus_Jakarta_Sans',sans-serif]
        ${
          settings.darkMode
            ? 'dark bg-slate-950 text-slate-100'
            : 'bg-slate-50 text-slate-900'
        }
      `}
    >
      {/* --------------------------------------------- */}
      {/* Universal Top Navbar */}
      {/* --------------------------------------------- */}

      <Navbar
        currentView={currentView}
        onNavigate={handleNavigate}
        user={user}
        onLogout={handleLogout}
        onLoginClick={() =>
          handleNavigate('login')
        }
      />

      {/* --------------------------------------------- */}
      {/* Main View */}
      {/* --------------------------------------------- */}

      <main className="flex-1">

        {/* ------------------------------------------- */}
        {/* Landing Page */}
        {/* ------------------------------------------- */}

        {currentView === 'landing' && (
          <LandingPage
            onNavigate={handleNavigate}
            onLoginClick={() =>
              handleNavigate('login')
            }
            onTrySample={() => {
              /*
               * This is intentionally a demo/sample result.
               * It does NOT call the Qwen backend.
               */

              setActiveResult(
                MOCK_INVOICE_TOPSTAR
              );

              setCurrentView('results');
            }}
          />
        )}

        {/* ------------------------------------------- */}
        {/* Login */}
        {/* ------------------------------------------- */}

        {currentView === 'login' && (
          <LoginPage
            onLoginSuccess={(loggedInUser) => {
              setUser(loggedInUser);

              setCurrentView('dashboard');
            }}
            onNavigate={handleNavigate}
          />
        )}

        {/* ------------------------------------------- */}
        {/* Register */}
        {/* ------------------------------------------- */}

        {currentView === 'register' && (
          <RegisterPage
            onRegisterSuccess={(registeredUser) => {
              setUser(registeredUser);

              setCurrentView('dashboard');
            }}
            onNavigate={handleNavigate}
          />
        )}

        {/* ------------------------------------------- */}
        {/* Dashboard */}
        {/* ------------------------------------------- */}

        {currentView === 'dashboard' && (
          <Dashboard
            user={user}
            history={history}
            onNavigate={handleNavigate}
            onSelectDocument={
              handleSelectDocument
            }
          />
        )}

        {/* ------------------------------------------- */}
        {/* Upload */}
        {/* ------------------------------------------- */}

        {currentView === 'upload' && (
          <UploadDocument
            onStartAnalysis={
              handleStartAnalysis
            }
            settings={settings}
          />
        )}

        {/* ------------------------------------------- */}
        {/* Processing */}
        {/* ------------------------------------------- */}

        {currentView === 'processing' && (
          <ProcessingView
            currentStageIndex={
              processingStage
            }
            fileName={
              processingFileName
            }
          />
        )}

        {/* ------------------------------------------- */}
        {/* Extraction Results */}
        {/* ------------------------------------------- */}

        {currentView === 'results' && (
          <ExtractionResults
            result={activeResult}
            onNavigate={handleNavigate}
            onAnalyzeAnother={() =>
              handleNavigate('upload')
            }
          />
        )}

        {/* ------------------------------------------- */}
        {/* History */}
        {/* ------------------------------------------- */}

        {currentView === 'history' && (
          <DocumentHistory
            history={history}
            onSelectDocument={
              handleSelectDocument
            }
            onNavigate={handleNavigate}
            onClearHistory={() =>
              setHistory([])
            }
          />
        )}

        {/* ------------------------------------------- */}
        {/* Document Details */}
        {/* ------------------------------------------- */}

        {currentView === 'details' &&
          selectedHistoryDoc && (
            <DocumentDetails
              document={
                selectedHistoryDoc
              }
              onNavigate={handleNavigate}
              onAnalyzeAgain={
                handleAnalyzeAgain
              }
            />
          )}

        {/* ------------------------------------------- */}
        {/* Profile Settings */}
        {/* ------------------------------------------- */}

        {currentView === 'profile' && (
          <ProfileSettings
            user={user}
            onUpdateUser={setUser}
            settings={settings}
            onUpdateSettings={
              setSettings
            }
            onLogout={handleLogout}
          />
        )}

      </main>
    </div>
  );
}