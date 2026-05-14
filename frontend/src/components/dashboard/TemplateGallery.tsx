/**
 * TemplateGallery — empty state on the dashboard.
 *
 * Three starter templates that seed a project with a description prompt.
 * The project canvas's seed input pre-fills with the description.
 */
import { motion } from 'framer-motion';
import { Sparkles, Layers, BookOpen, Compass } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { staggerContainer, fadeRise, depthHue } from '../../styles/motion';

interface Template {
  id: string;
  Icon: LucideIcon;
  title: string;
  description: string;
  prompt: string;
  hue: number;
}

const TEMPLATES: Template[] = [
  {
    id: 'product-brainstorm',
    Icon: Sparkles,
    title: 'Product brainstorm',
    description: 'Generate diverse feature directions for a product problem.',
    prompt: 'How might we improve the onboarding flow so new users reach an "aha" moment in under 60 seconds?',
    hue: depthHue(0),
  },
  {
    id: 'essay-outline',
    Icon: BookOpen,
    title: 'Essay outline',
    description: 'Branch into multiple thesis angles for an argument.',
    prompt: 'Write a 4-paragraph essay on whether large language models are creative thinkers or pattern matchers.',
    hue: depthHue(1),
  },
  {
    id: 'research-questions',
    Icon: Compass,
    title: 'Research questions',
    description: 'Map the question space around an emerging topic.',
    prompt: 'What are the most important open questions in representation engineering for diversity in LLMs?',
    hue: depthHue(2),
  },
];

interface Props {
  onUseTemplate: (title: string, prompt: string) => void;
}

export function TemplateGallery({ onUseTemplate }: Props) {
  return (
    <section className="mt-12">
      <div className="flex items-baseline justify-between mb-4">
        <h2
          className="font-display text-ink-0 text-[18px] tracking-tight"
          style={{ fontVariationSettings: '"wght" 480, "opsz" 32' }}
        >
          Start from a template
        </h2>
        <span className="inline-flex items-center gap-1.5 text-[11px] text-ink-2">
          <Layers size={11} /> {TEMPLATES.length} starters
        </span>
      </div>
      <motion.div
        variants={staggerContainer}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
        className="grid grid-cols-1 md:grid-cols-3 gap-3"
      >
        {TEMPLATES.map(({ id, Icon, title, description, prompt, hue }) => (
          <motion.button
            key={id}
            variants={fadeRise}
            onClick={() => onUseTemplate(title, prompt)}
            className="group text-left p-4 rounded-xl bg-paper-1 border border-[var(--ink-line)] hover:bg-paper-2 hover:border-[var(--ink-line-bold)] transition-all"
            style={{ '--gen-hue': hue } as React.CSSProperties}
          >
            <div className="flex items-center gap-2 mb-3">
              <span
                className="h-8 w-8 rounded-md flex items-center justify-center"
                style={{ background: 'oklch(0.95 0.04 var(--gen-hue) / 0.7)', color: 'oklch(0.45 0.16 var(--gen-hue))' }}
              >
                <Icon size={14} />
              </span>
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: 'var(--gen)' }} />
            </div>
            <h3
              className="font-display text-ink-0 text-[15px] tracking-tight mb-1.5 group-hover:text-brand transition-colors"
              style={{ fontVariationSettings: '"wght" 520, "opsz" 24' }}
            >
              {title}
            </h3>
            <p className="text-ink-1 text-[12.5px] leading-relaxed mb-3">{description}</p>
            <div className="text-[11px] text-ink-3 line-clamp-2 italic font-display" style={{ fontVariationSettings: '"wght" 400, "opsz" 14' }}>
              "{prompt}"
            </div>
          </motion.button>
        ))}
      </motion.div>
    </section>
  );
}
