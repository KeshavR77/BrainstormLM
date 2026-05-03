/**
 * App.tsx — Main application shell with hash-based routing.
 *
 * Routes:
 *   #/             → Landing (Corpus marketing + hero idea input)
 *   #/app          → Dashboard (project listing)
 *   #/project/:id  → Canvas (brainstorm workspace)
 */

import { useCallback, useEffect, useState } from 'react';
import './index.css';
import './components/nodes/nodes.css';
import './pages/pages.css';
import './pages/landing-scroll.css';

import { Landing } from './pages/Landing';
import { Dashboard } from './pages/Dashboard';
import { Canvas } from './pages/Canvas';

type Route =
  | { page: 'landing' }
  | { page: 'dashboard' }
  | { page: 'canvas'; projectId: string };

function parseRoute(): Route {
  const hash = window.location.hash || '#/';
  const projectMatch = hash.match(/^#\/project\/(.+)$/);
  if (projectMatch) {
    return { page: 'canvas', projectId: projectMatch[1] };
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

  // Lock body scroll only for the Canvas workspace.
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
  const goToDashboard = openDashboard;

  switch (route.page) {
    case 'canvas':
      return <Canvas projectId={route.projectId} onBack={goToDashboard} />;
    case 'dashboard':
      return <Dashboard onOpenProject={openProject} />;
    case 'landing':
    default:
      return (
        <Landing
          onStartProject={openProject}
          onOpenDashboard={openDashboard}
        />
      );
  }
}
