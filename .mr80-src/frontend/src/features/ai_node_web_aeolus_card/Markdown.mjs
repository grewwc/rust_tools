import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function Markdown({ content }) {
  return React.createElement('div', { className: 'contract-markdown' },
    React.createElement(ReactMarkdown, {
      remarkPlugins: [remarkGfm],
      skipHtml: true,
      components: {
        a: ({ node, ...props }) => React.createElement('a', {
          ...props, target: '_blank', rel: 'noopener noreferrer',
        }),
      },
    }, content));
}
