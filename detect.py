import cv2
from pathlib import Path
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('image_name', type=str,
                    help='An argument representing image in input folder')
args = parser.parse_args()

cur = Path('.')

INPUT_FILE= str(cur / 'input' / args.image_name)
OUTPUT_FILE= str(cur / 'output' /  'predicted.jpg')
LABELS_FILE= str( cur / 'data'/ 'cards.names')
CONFIG_FILE= str(cur / 'yolov4' / 'yolo-obj.cfg')
WEIGHTS_FILE= str(cur / 'yolov4' /'yolo-obj_last.weights')
CONFIDENCE_THRESHOLD=0.3

def determine_hand(hand):
		card_value = dict(zip('2 3 4 5 6 7 8 9 T J Q K A'.split(), range(14)))

		cards = []
		suits = []

		for card in hand.split():
			c, s = list(card)
			cards.append(c)
			suits.append(s)

		max_suit = max([suits.count(a) for a in suits])
		same_cards = sorted([cards.count(a) for a in set(cards)])
		card_nums = sorted([card_value[a] for a in cards])

		def is_straight(cv):
			diff = cv[-1] - cv[0]
			if diff == 4:
				return True
			elif diff == 12:
				if cv[-2] - cv[0] == 3:
					return True
			return False

		# We have our flushes in here. Any less suits and we don't care.
		if max_suit == 5:
			if is_straight(card_nums):
				if card_nums[0] == 8: # ROYAL FLUSH!!!
					return "Royal Flush"
				return "Straight Flush"
			return "Flush"

		# Checking in on our same cards
		# With a length of two we either have two pair or a full house
		elif len(same_cards) == 2:
			if max(same_cards) == 4: # Four of a Kind
				return "Four of a Kind"
			elif max(same_cards) == 3: # Full House
				return "Full House"
		elif len(same_cards) == 3:
			if max(same_cards) == 3: # Three of a kind
				return "Three of a Kind"
			else: # Two pair
				return "Two Pair"
		elif len(same_cards) == 4:
			return "One Pair"
		else: # Garbage hand most likely. But maybe a straight!
			if is_straight(card_nums):
				return "Straight"
			return "High Card"

def main():
	with open(LABELS_FILE, 'r') as f:
		classes = f.read().splitlines()
	
	for i in range(len(classes)):
		c = classes[i]
		if '10' in c:
			classes[i] = c.replace('10', 'T')
	

	net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

	image = cv2.imread(INPUT_FILE)
	(H, W) = image.shape[:2]


	
	model = cv2.dnn_DetectionModel(net)
	model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
	
	classIds, scores, boxes = model.detect(image, confThreshold=0.3, nmsThreshold=0.4)

	# print(classes)


	cardsOnBoard = {}
	i=1
	for  (classId, score, box) in zip(classIds, scores, boxes):
		card = classes[classId]

		if card in cardsOnBoard:
			oldScore = cardsOnBoard[card][1]
			if oldScore > score:
				boxY = cardsOnBoard[card][0]
				cardsOnBoard[card] = (boxY, score)
			continue 
		else:
			i+1
			cardsOnBoard[card] = (box[1], score)
		cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),color=(0, 255, 0), thickness=2)
		text = '%s: %.2f' % (card, score)
		cv2.putText(image, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,color=(0, 255, 0), thickness=2)
		
	
	cardsOnBoard = [x[0] for x in sorted(cardsOnBoard.items(), key=lambda item: item[1][0])]

	numCards = len(cardsOnBoard)


	if numCards > 5 or numCards < 5:
		print('Num cards ',numCards)
		print('Number of cards on board should be euqal to 5')
		return


	cards = " ".join(cardsOnBoard).upper()
	handRank = determine_hand(cards)
	print(cardsOnBoard)
	print(f"Your hand rank: {handRank}\n")
	cv2.putText(image, f"Your hand rank: {handRank}", (W//4, H-10), cv2.FONT_HERSHEY_SIMPLEX, 1,color=(0, 255, 0), thickness=2)
	cv2.imshow('Image', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite(OUTPUT_FILE, image)
	
    

if __name__ == "__main__":
    main()