/**
 * App.tsx — Main application shell with hash-based routing.
 *
 * Routes:
 *   #/             → Corpus 1 Landing (marketing + demo CTA)
 *   #/demo         → Demo (Conceptial vs Gemini-direct — headline 2-way)
 *   #/frontier     → Frontier (3-arm cross-vendor: GPT-4o / Claude / Conceptial)
 *   #/app          → Dashboard (project listing)
 *   #/project/:id  → Canvas (brainstorm workspace)
 */

import { useCallback, useEffect, useState } from 'react';
import './index.css';
import './components/nodes/nodes.css';

import { Corpus1Landing } from './pages/Corpus1Landing';
import { Dashboard } from './pages/Dashboard';
import { Canvas } from './pages/Canvas';
import { Demo } from './pages/Demo';
import { Frontier } from './pages/Frontier';
import { IntroSplash } from './components/IntroSplash';

type Route =
  | { page: 'landing' }
  | { page: 'demo' }
  | { page: 'frontier' }
  | { page: 'dashboard' }
  | { page: 'canvas'; projectId: string };

function parseRoute(): Route {
  const hash = window.location.hash || '#/';
  const projectMatch = hash.match(/^#\/project\/(.+)$/);
  if (projectMatch) {
    return { page: 'canvas', projectId: projectMatch[1] };
  }
  if (hash === '#/demo' || hash.startsWith('#/demo/')) {
    return { page: 'demo' };
  }
  if (hash === '#/frontier' || hash.startsWith('#/frontier/')) {
    return { page: 'frontier' };
  }
  if (hash === '#/app' || hash.startsWith('#/app/')) {
    return { page: 'dashboard' };
  }
  return { page: 'landing' };
}

export default function App() {
  const [route, setRoute] = useState<Route>(parseRoute);

  const [showIntro, setShowIntro] = useState(
    () => parseRoute().page === 'landing' && !sessionStorage.getItem('conceptial-intro-seen'),
  );

  useEffect(() => {
    if (route.page === 'landing' && !sessionStorage.getItem('conceptial-intro-seen')) {
      if (!showIntro) setShowIntro(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [route.page]);

  useEffect(() => {
    const handleHash = () => setRoute(parseRoute());
    window.addEventListener('hashchange', handleHash);
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);

  useEffect(() => {
    const body = document.body;
    if (route.page === 'canvas') {
      body.classList.add('is-canvas');
    } else {
      body.classList.remove('is-canvas');
    }
    return () => body.classList.remove('is-canvas');
  }, [route.page]);

  const navigateTo = useCallback((path: string) => {
    window.location.hash = path;
  }, []);

  const openProject = useCallback(
    (id: string) => navigateTo(`/project/${id}`),
    [navigateTo],
  );

  const openDashboard = useCallback(() => navigateTo('/app'), [navigateTo]);
  const openDemo = useCallback(() => navigateTo('/demo'), [navigateTo]);
  const openLanding = useCallback(() => navigateTo('/'), [navigateTo]);

  const page = (() => {
    switch (route.page) {
      case 'canvas':
        return <Canvas key={route.projectId} projectId={route.projectId} onBack={openDashboard} />;
      case 'demo':
        return <Demo onBack={openLanding} onOpenWorkspace={openDashboard} />;
      case 'frontier':
        return <Frontier onBack={openLanding} onOpenWorkspace={openDashboard} />;
      case 'dashboard':
        return <Dashboard onOpenProject={openProject} />;
      case 'landing':
      default:
        return (
          <Corpus1Landing
            onLaunchDemo={openDemo}
            onOpenWorkspace={openDashboard}
          />
        );
    }
  })();

  return (
    <>
      {page}
      {showIntro && (
        <IntroSplash
          onDone={() => {
            sessionStorage.setItem('conceptial-intro-seen', '1');
            setShowIntro(false);
          }}
        />
      )}
    </>
  );
}
