/**
 * App.tsx — Main application shell with hash-based routing.
 *
 * Routes:
 *   #/             → Corpus 1 Landing (marketing + demo CTA)
 *   #/gauntlet     → Gauntlet (live steered generation demo)
 *   #/demo         → Demo (Conceptial vs Gemini-direct — headline 2-way)
 *   #/frontier     → Frontier (3-arm cross-vendor: GPT-4o / Claude / Conceptial)
 *   #/control      → Control (Gemini-direct alone, 16 outputs)
 *   #/app          → Dashboard (project listing)
 *   #/project/:id  → Canvas (brainstorm workspace)
 */

import { useCallback, useEffect, useState } from 'react';
import './index.css';
import './components/nodes/nodes.css';

import { Corpus1Landing } from './pages/Corpus1Landing';
import { Gauntlet } from './pages/Gauntlet';
import { Dashboard } from './pages/Dashboard';
import { Canvas } from './pages/Canvas';
import { Demo } from './pages/Demo';
import { Frontier } from './pages/Frontier';
import { Control } from './pages/Control';
import { IntroSplash } from './components/IntroSplash';

type Route =
  | { page: 'landing' }
  | { page: 'gauntlet' }
  | { page: 'demo' }
  | { page: 'frontier' }
  | { page: 'control' }
  | { page: 'dashboard' }
  | { page: 'canvas'; projectId: string };

function parseRoute(): Route {
  const hash = window.location.hash || '#/';
  const projectMatch = hash.match(/^#\/project\/(.+)$/);
  if (projectMatch) {
    return { page: 'canvas', projectId: projectMatch[1] };
  }
  if (hash === '#/gauntlet' || hash.startsWith('#/gauntlet/')) {
    return { page: 'gauntlet' };
  }
  if (hash === '#/demo' || hash.startsWith('#/demo/')) {
    return { page: 'demo' };
  }
  if (hash === '#/frontier' || hash.startsWith('#/frontier/')) {
    return { page: 'frontier' };
  }
  if (hash === '#/control' || hash.startsWith('#/control/')) {
    return { page: 'control' };
  }
  if (hash === '#/app' || hash.startsWith('#/app/')) {
    return { page: 'dashboard' };
  }
  return { page: 'landing' };
}

export default function App() {
  const [route, setRoute] = useState<Route>(parseRoute);

  /* Intro splash — plays on landing-route fresh page loads.
     Skipped when navigating back from another in-app route, and skipped once it has played
     in the current tab session (so client-side hash changes don't replay it). */
  const [showIntro, setShowIntro] = useState(
    () => parseRoute().page === 'landing' && !sessionStorage.getItem('conceptial-intro-seen'),
  );

  /* Reset showIntro to true whenever the user explicitly reloads the landing page
     by re-evaluating sessionStorage on hashchange to landing. (Doesn't fire on
     initial mount — that's the useState initializer above.) */
  useEffect(() => {
    if (route.page === 'landing' && !sessionStorage.getItem('conceptial-intro-seen')) {
      // covers HMR-state-preservation edge case
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
  const openGauntlet = useCallback(() => navigateTo('/gauntlet'), [navigateTo]);
  const openDemo = useCallback(() => navigateTo('/demo'), [navigateTo]);
  const openLanding = useCallback(() => navigateTo('/'), [navigateTo]);

  const page = (() => {
    switch (route.page) {
      case 'canvas':
        return <Canvas key={route.projectId} projectId={route.projectId} onBack={openDashboard} />;
      case 'gauntlet':
        return <Gauntlet onBack={openLanding} />;
      case 'demo':
        return <Demo onBack={openLanding} onOpenWorkspace={openDashboard} />;
      case 'frontier':
        return <Frontier onBack={openLanding} onOpenWorkspace={openDashboard} />;
      case 'control':
        return <Control onBack={openLanding} onOpenDemo={openDemo} onOpenWorkspace={openDashboard} />;
      case 'dashboard':
        return <Dashboard onOpenProject={openProject} />;
      case 'landing':
      default:
        return (
          <Corpus1Landing
            onLaunchGauntlet={openGauntlet}
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
