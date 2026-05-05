/**
 * App.tsx — Main application shell with hash-based routing.
 *
 * Routes:
 *   #/             → Corpus 1 Landing (marketing + demo CTA)
 *   #/gauntlet     → Gauntlet (live steered generation demo)
 *   #/app          → Dashboard (project listing)
 *   #/project/:id  → Canvas (brainstorm workspace)
 */

import { useCallback, useEffect, useState } from 'react';
import './index.css';
import './components/nodes/nodes.css';
import './pages/pages.css';
import './pages/landing-scroll.css';
import './pages/corpus1.css';
import './pages/gauntlet.css';

import { Corpus1Landing } from './pages/Corpus1Landing';
import { Gauntlet } from './pages/Gauntlet';
import { Dashboard } from './pages/Dashboard';
import { Canvas } from './pages/Canvas';

type Route =
  | { page: 'landing' }
  | { page: 'gauntlet' }
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
  if (hash === '#/app' || hash.startsWith('#/app/')) {
    return { page: 'dashboard' };
  }
  return { page: 'landing' };
}

export default function App() {
  const [route, setRoute] = useState<Route>(parseRoute);

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
  const openLanding = useCallback(() => navigateTo('/'), [navigateTo]);

  switch (route.page) {
    case 'canvas':
      return <Canvas key={route.projectId} projectId={route.projectId} onBack={openDashboard} />;
    case 'gauntlet':
      return <Gauntlet onBack={openLanding} />;
    case 'dashboard':
      return <Dashboard onOpenProject={openProject} />;
    case 'landing':
    default:
      return (
        <Corpus1Landing
          onLaunchGauntlet={openGauntlet}
          onOpenWorkspace={openDashboard}
        />
      );
  }
}
